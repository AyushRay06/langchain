import { ChatGoogleGenerativeAI } from "@langchain/google-genai"

import * as dotenv from "dotenv"
import { ChatPromptTemplate } from "@langchain/core/prompts"

dotenv.config()

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-pro",
  temperature: 0.5,
  maxRetries: 3,
})

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are a helpful assistant that translates {input_language} to {output_language}.",
  ],
  ["human", "{input}"],
])

const chain = prompt.pipe(llm)

const response = await chain.invoke({
  input_language: "English",
  output_language: "hindi",
  input: "I love programming.",
})

console.log(response.content)
