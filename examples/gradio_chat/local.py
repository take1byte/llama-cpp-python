import uuid

import gradio as gr
from guard import guard, logging_utils
from guard.constants import (
    INPUT_FORMAT_MSG,
    OUT_OF_SCOPE_MSG,
    UNAUTHORIZED_TEXT_IN_PROMPT_MSG,
)
from guard.prompt_handler import (
    rewrite,
    secure_against_prompt_injection,
    split_instructions_and_data,
)
from guard.response_handler import response_generator

import llama_cpp
import llama_cpp.llama_tokenizer

llama = llama_cpp.Llama.from_pretrained(
    repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="*Q4_K_M.gguf",
    n_ctx=4096,
    verbose=False,
)

model = "gpt-3.5-turbo"

convo_id = str(uuid.uuid4())[:8]
logger = logging_utils.build_logger("convo_log", f"convo_log_{convo_id}.log")

SECURED_AGAINST_PROMPT_INJECTIONS = True
SECURED_BY_POLICY = False

WRONG_FORMAT_MSG = "User message in wrong format"


def secured_against_prompt_injections_predict(message, history):
    messages = []

    instr, data = split_instructions_and_data(message)

    logger.info(f"user message:{message}")
    logger.info(f"user instruction:{instr}; user data:{data}")
    logger.info(f"history: {history}")

    if instr is None:
        for text in response_generator(INPUT_FORMAT_MSG):
            yield text
    else:
        secure_instr, secure_data, authoring_hint = secure_against_prompt_injection(instr, data)
        logger.info(f"\nSecure instruction: {secure_instr}\nSecure data: {secure_data}")

        if authoring_hint is None:
            message = secure_instr if data is None else f"{secure_instr} {secure_data}"

            for record in history:
                if record.get("role") == "user":
                    messages.append({"role": "user", "content": record.get("content")})
                elif record.get("role") == "assistant":
                    messages.append({"role": "assistant", "content": record.get("content")})

            messages.append({"role": "user", "metadata": {"title": None}, "content": message})
            response = llama.create_chat_completion_openai_v1(
                model=model, messages=messages, stream=True
            )
            text = ""
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    text += content
                    yield text
                elif text != "":
                    logger.info(f"Response: {text}")
        else:
            logger.info(f"Response: {authoring_hint}")
            for text in response_generator(authoring_hint):
                yield text


def secured_by_policy_predict(message, history, principal="guest@domain.com"):
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
            logger.info("is_permitted: True")

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
            logger.info("is_permitted: False")
            for text in response_generator(OUT_OF_SCOPE_MSG):
                yield text


def predict(message, history):
    messages = []

    logger.info(f"user message:{message}")

    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

    messages.append({"role": "user", "content": message})
    response = llama.create_chat_completion_openai_v1(model=model, messages=messages, stream=True)
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

with gr.Blocks(js=js, css=css) as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(
        label=(
            "Please input your prompt as I: [your instruction or question]"
            "D: [(optional) data, e.g., document]"
        )
    )
    clear = gr.Button("Clear")

    def user(user_message, history: list):
        return "", history + [
            {"role": "user", "content": user_message, "original_message": user_message}
        ]

    def bot(history: list):
        messages = []

        if len(history) > 0 and history[-1]["role"] == "user":
            user_message = history[-1]["content"]
        else:
            user_message = ""

        instr, data = split_instructions_and_data(user_message)

        logger.info(f"user message:{user_message}")
        logger.info(f"user instruction:{instr}; user data:{data}")
        logger.info(f"history: {history}")

        history.append({"role": "assistant", "content": ""})

        if instr is None:
            history[-2]["content"] = WRONG_FORMAT_MSG
            for text in response_generator(INPUT_FORMAT_MSG):
                history[-1]["content"] = text
                yield history
        else:
            secure_instr, secure_data, authoring_hint = secure_against_prompt_injection(instr, data)
            logger.info(f"\nSecure instruction: {secure_instr}\nSecure data: {secure_data}")

            if authoring_hint is None:
                message = secure_instr if data is None else f"{secure_instr} {secure_data}"
                history[-2]["content"] = message

                for record in history:
                    if record.get("role") == "user":
                        messages.append(
                            {
                                "role": "user",
                                "content": record.get("content"),
                            }
                        )
                    elif record.get("role") == "assistant":
                        messages.append(
                            {
                                "role": "assistant",
                                "content": record.get("content"),
                            }
                        )

                logger.info(f"messages: {messages}")
                response = llama.create_chat_completion_openai_v1(
                    model=model, messages=messages, stream=True
                )

                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        history[-1]["content"] += content
                        yield history
                    elif history[-1]["content"] != "":
                        logger.info(f"Response: {history[-1]["content"]}")
            else:
                logger.info(f"Response: {authoring_hint}")
                history[-2]["content"] = UNAUTHORIZED_TEXT_IN_PROMPT_MSG
                for text in response_generator(authoring_hint):
                    history[-1]["content"] = text
                    yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch()
