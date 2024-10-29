import llama_cpp
import llama_cpp.llama_tokenizer
import uuid
import gradio as gr
from time import sleep
from guard import guard
from guard import logging_utils
from guard.constants import INPUT_FORMAT_MSG, OUT_OF_SCOPE_MSG
from guard.prompt_handler import split_instructions_and_data, rewrite
from guard.response_handler import response_generator
from llm_guard.input_scanners import BanTopics

llama = llama_cpp.Llama.from_pretrained(
    repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="*Q4_K_M.gguf",
    n_ctx=1024,
    verbose=False,
)

model = "gpt-3.5-turbo"

convo_id = str(uuid.uuid4())[:8]
logger = logging_utils.build_logger("convo_log", f"convo_log_{convo_id}.log")

SECURED = False
LLM_GUARD = True


def secured_predict(message, history, principal="guest@domain.com"):
    messages = []

    instr, data = split_instructions_and_data(message)

    logger.info(f"user message:{message}")
    logger.info(f"user instruction:{instr}; user data:{data}")

    if instr is None:
        for text in response_generator(INPUT_FORMAT_MSG):
            yield text
    else:
        rewritten_instr = rewrite(instr)
        logger.info(f"rewritten user instruction: {rewritten_instr}")

        message = rewritten_instr if data is None else f"{rewritten_instr} {data}"

        for user_message, assistant_message in history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})

        if guard.is_permitted(rewritten_instr, principal=principal):
            logger.info(f"is_permitted: True")

            messages.append({"role": "user", "content": message})
            response = llama.create_chat_completion_openai_v1(
                model=model, messages=messages, stream=True
            )
            text = ""
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    text += content
                    yield text
        else:
            logger.info(f"is_permitted: False")
            for text in response_generator(OUT_OF_SCOPE_MSG):
                yield text


def llm_guard_predict(message, history):
    messages = []

    logger.info(f"user message:{message}")

    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

    scanner = BanTopics(topics=["violence"], threshold=0.5)
    sanitized_prompt, is_valid, risk_score = scanner.scan(message)
    logger.info(
        f"sanitized_prompt: {sanitized_prompt}; is_valid: {is_valid}; risk_score: {risk_score}"
    )

    if is_valid:
        messages.append({"role": "user", "content": sanitized_prompt})
        response = llama.create_chat_completion_openai_v1(
            model=model, messages=messages, stream=True
        )
        text = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                text += content
                yield text
    else:
        for text in response_generator(OUT_OF_SCOPE_MSG):
            yield text


def predict(message, history):
    messages = []

    logger.info(f"user message:{message}")

    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

    messages.append({"role": "user", "content": message})
    response = llama.create_chat_completion_openai_v1(
        model=model, messages=messages, stream=True
    )
    text = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            text += content
            yield text


js = """function () {
  gradioURL = window.location.href
  if (!gradioURL.endsWith('?__theme=dark')) {
    window.location.replace(gradioURL + '?__theme=dark');
  }
}"""

css = """
footer {
    visibility: hidden;
}
full-height {
    height: 100%;
}
label.svelte-1b6s6s {visibility: hidden}
"""

with gr.Blocks(theme=gr.themes.Soft(), js=js, css=css, fill_height=True) as demo:
    title = "Secured Chat" if SECURED else "Unsecured Chat"
    examples = (
        [
            "I: summarize the document D: [document text]",
            "I: write an article about cybersecurity",
        ]
        if SECURED
        else [
            "summarize the document: [document text]",
            "write an article about cybersecurity",
        ]
    )

    if SECURED:
        predict_fn = secured_predict
    elif LLM_GUARD:
        predict_fn = llm_guard_predict
    else:
        predict_fn = predict

    gr.ChatInterface(
        predict_fn,
        fill_height=True,
        examples=None,
        title=title,
    )


if __name__ == "__main__":
    demo.queue().launch()
