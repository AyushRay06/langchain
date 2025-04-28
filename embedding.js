import { ChatPromptTemplate } from "@langchain/core/prompts"
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"

import * as dotenv from "dotenv"

dotenv.config()

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-pro",
  temperature: 0.5,
  maxRetries: 3,
})

const prompt = ChatPromptTemplate.fromMessages([
    ["system", ""]
])
