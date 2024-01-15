import argparse
import hashlib
import json
import os
import time
from threading import Thread

import gradio as gr
import torch
from llava_phi.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                 DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava_phi.conversation import (SeparatorStyle, conv_templates,
                                    default_conversation)
from llava_phi.mm_utils import (KeywordsStoppingCriteria, load_image_from_base64,
                                process_images, tokenizer_image_token)
from llava_phi.model.builder import load_pretrained_model
from transformers import TextIteratorStreamer

print(gr.__version__)

block_css = """

#buttons button {
    min-width: min(120px,100%);
}
"""
title_markdown = ("""
# LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model
[[Code](https://github.com/zhuyiche/llava-phi)] | 📚 [[Paper](https://arxiv.org/pdf/2401.02330)]
""")
tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")
learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://huggingface.co/microsoft/phi-2) of Phi-2. Please contact us if you find any potential violation.
""")
ack_markdown = ("""
### Acknowledgement
The template for this web demo is from [LLaVA](https://github.com/haotian-liu/LLaVA), and we are very grateful to LLaVA for their open source contributions to the community!
""")


def regenerate(state, image_process_mode):
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None)


def clear_history():
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None)


def add_text(state, text, image, image_process_mode):
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None)

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            # text = '<Image><image></Image>' + text
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None)


def load_demo():
    state = default_conversation.copy()
    return state


@torch.inference_mode()
def get_response(params):
    prompt = params["prompt"]
    ori_prompt = prompt
    images = params.get("images", None)
    num_image_tokens = 0
    if images is not None and len(images) > 0:
        if len(images) > 0:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError(
                    "Number of images does not match number of <image> tokens in prompt")

            images = [load_image_from_base64(image) for image in images]
            images = process_images(images, image_processor, model.config)

            if type(images) is list:
                images = [image.to(model.device, dtype=torch.float16)
                          for image in images]
            else:
                images = images.to(model.device, dtype=torch.float16)

            replace_token = DEFAULT_IMAGE_TOKEN
            if getattr(model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            num_image_tokens = prompt.count(
                replace_token) * model.get_vision_tower().num_patches
        else:
            images = None
        image_args = {"images": images}
    else:
        images = None
        image_args = {}

    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_context_length = getattr(
        model.config, 'max_position_embeddings', 2048)
    max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
    stop_str = params.get("stop", None)
    do_sample = True if temperature > 0.001 else False

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(
        keywords, tokenizer, input_ids)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

    max_new_tokens = min(max_new_tokens, max_context_length -
                         input_ids.shape[-1] - num_image_tokens)

    if max_new_tokens < 1:
        yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.",
                          "error_code": 0}).encode() + b"\0"
        return

    # local inference
    thread = Thread(target=model.generate, kwargs=dict(
        inputs=input_ids,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        stopping_criteria=[stopping_criteria],
        use_cache=True,
        **image_args
    ))
    thread.start()

    generated_text = ori_prompt
    for new_text in streamer:
        generated_text += new_text
        if generated_text.endswith(stop_str):
            generated_text = generated_text[:-len(stop_str)]
        yield json.dumps({"text": generated_text, "error_code": 0}).encode()


def http_bot(state, temperature, top_p, max_new_tokens):
    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot())
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "phi" in model_name.lower():
            template_name = "phi-2_v0"
        else:
            template_name = "phi-2_v0"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest()
                      for image in all_images]

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot())

    # for stream
    output = get_response(pload)
    for chunk in output:
        if chunk:
            data = json.loads(chunk.decode())
            if data["error_code"] == 0:
                output = data["text"][len(prompt):].strip()
                state.messages[-1][-1] = output + "▌"
                yield (state, state.to_gradio_chatbot())
            else:
                output = data["text"] + \
                         f" (error_code: {data['error_code']})"
                state.messages[-1][-1] = output
                yield (state, state.to_gradio_chatbot())
                return
            time.sleep(0.03)

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot())


def build_demo():
    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LLaVA-Phi", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()
        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row(elem_id="Model ID"):
                    gr.Dropdown(
                        choices=['LLaVA-Phi-3B'],
                        value='LLaVA-Phi-3B',
                        interactive=True,
                        label='Model ID',
                        container=False)
                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/extreme_ironing.jpg",
                     "What is unusual about this image?"],
                    [f"{cur_dir}/examples/waterview.jpg",
                     "What are the things I should be cautious about when I visit here?"],
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=False) as _:
                    temperature = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature", )
                    top_p = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P", )
                    max_output_tokens = gr.Slider(
                        minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens", )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label="LLaVA-Phi Chatbot", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as _:
                    regenerate_btn = gr.Button(
                        value="🔄  Regenerate", interactive=True)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=True)

        gr.Markdown(tos_markdown)
        gr.Markdown(learn_more_markdown)
        gr.Markdown(ack_markdown)

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False
        ).then(
            http_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot]
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox],
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False
        ).then(
            http_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot]
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False
        ).then(
            http_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot]
        )

        demo.load(
            load_demo,
            None,
            [state],
            queue=False
        )
    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", default=True)
    parser.add_argument("--model-path", type=str,
                        default="checkpoints/llavaPhi-v0-3b-finetune")
    parser.add_argument("--model-name", type=str,
                        default="llavaPhi-v0-3b")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, args.model_name, False, False)
    demo = build_demo()
    demo.queue()
    demo.launch(server_name=args.host,
                server_port=args.port,
                share=args.share)
