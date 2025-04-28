import { ChatGoogleGenerativeAI } from "@langchain/google-genai"

import * as dotenv from "dotenv"
import { ChatPromptTemplate } from "@langchain/core/prompts"

dotenv.config()

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-pro",
  temperature: 0.5,
  maxRetries: 3,
})

const prompt = ChatPromptTemplate.fromTemplate(
  "You are a Comicbook nerd, Tell me interesting things about the {input}."
)

const chain = prompt.pipe(llm)

const response = await chain.invoke({
  input: "Ironman",
})

console.log(response.content)
