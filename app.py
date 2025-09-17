import getpass
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Inisialisasi API Key Groq
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

# Inisialisasi Model
model = init_chat_model(
    "llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0
)

# Prompt Template
system_prompt = SystemMessagePromptTemplate.from_template(
    "Kamu adalah AI asisten. Jangan ulangi informasi pribadi user lebih dari sekali per respon, kecuali user bertanya terkait hal tersebut. Jawablah dengan padat dan jelas."
)

chat_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Tambah Memory
memory = ConversationTokenBufferMemory(
    llm=model,
    max_token_limit=2000,
    return_messages=True,
    memory_key="history" 
)


# Inisialisasi ConversationChain
conversation = ConversationChain(
    llm=model,
    memory=memory,
    prompt=chat_prompt,
    # verbose=True
)

# Flask App
app = Flask(__name__)
CORS(app)

@app.route("/llm", methods=["POST"])
def llm_api():
    try:
        data = request.json
        user_input = data.get("prompt", "")

        if not user_input:
            return jsonify({"error": "Prompt is required"}), 400

        response = conversation.run(input=user_input)

        messages = memory.buffer_as_messages

        conversation_data = []
        for msg in messages:
            if msg.type == "human":
                role = "Human"
            elif msg.type == "ai":
                role = "AI"
            else:
                role = "System"
            conversation_data.append({"role": role, "content": msg.content})

        return jsonify({
            "response": response,
            "conversation": conversation_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/llm/clear_memory", methods=["POST"])
def clear_memory_endpoint():
    try:
        memory.clear()
        return jsonify({"message": "Memory Empty"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask
if __name__ == "__main__":
    app.run(debug=True)