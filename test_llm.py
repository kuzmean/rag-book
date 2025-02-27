#!/usr/bin/env python3
import argparse
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

from langchain_openai import ChatOpenAI

def main():
    parser = argparse.ArgumentParser(description="Translate English to French using ChatOpenAI")
    parser.add_argument("text", nargs="?", default="I love programming.", help="Text to translate")
    args = parser.parse_args()
    
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        base_url="https://api.proxyapi.ru/openai/v1",
    )
    
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", args.text),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)

if __name__ == "__main__":
    main()