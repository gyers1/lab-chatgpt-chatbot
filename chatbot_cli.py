from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def main() -> None:
    load_dotenv()
    model = "gpt-4o-mini"
    system_prompt = "당신은 친절한 AI 어시스턴트입니다. 모르면 '확인 필요'라고만 답하되 아는 범위는 먼저 설명."
    llm = ChatOpenAI(model=model, temperature=0.3)
    history: list[dict[str, str]] = [
        {"role": "user", "content": "내 이름은 정태준이야."},
        {"role": "assistant", "content": "네, 정태준님 반가워요!"},
    ]

    def reset() -> None:
        history.clear()
        print("(대화 초기화 완료)")

    def chat(message: str) -> str:
        history.append({"role": "user", "content": message})

        # Notebook의 system + history -> LangChain 메시지 변환을 그대로 사용한다.
        msgs = [SystemMessage(content=system_prompt)]
        for item in history:
            cls = HumanMessage if item["role"] == "user" else AIMessage
            msgs.append(cls(content=item["content"]))

        resp = llm.invoke(msgs)
        answer = resp.content or ""
        history.append({"role": "assistant", "content": answer})
        return answer

    print("=== CLI 멀티턴 챗봇 (LangChain) ===")
    print("/reset - 대화 초기화")
    print("/history - 현재 히스토리 보기")
    print("/exit, /quit, /q - 종료")
    print()
    print("대화를 시작하려면 메시지를 입력하세요.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("종료")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"/exit", "/quit", "/q"}:
            print("종료")
            break
        if lowered == "/reset":
            reset()
            continue
        if lowered == "/history":
            if not history:
                print("(현재 히스토리가 없습니다)")
                continue
            for idx, message in enumerate(history, start=1):
                role = message["role"]
                content = message["content"]
                print(f"[{idx}] {role}: {content}")
            continue

        answer = chat(user_input)
        print("AI:", answer)


if __name__ == "__main__":
    main()
