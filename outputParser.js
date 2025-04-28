import { ChatPromptTemplate } from "@langchain/core/prompts"
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"
import {
  CommaSeparatedListOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers"

import { StructuredOutputParser } from "langchain/output_parsers"

import * as dotenv from "dotenv"
import { z } from "zod"
dotenv.config()

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-pro",
  temperature: 0.5,
  maxRetries: 3,
})

// I have create function so that i can run different output Parser in the same file, by simply calling them.
// We dont need to do this in actual project

// string parser
async function StringParse() {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Generate a joke based on a word provided by the user."],
    ["human", "{input}"],
  ])

  const parser = new StringOutputParser()

  const chain = prompt.pipe(llm).pipe(parser)
  return await chain.invoke({
    input: "dog",
  })
}

// list parser
async function ListParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Provide 5 superpower seperated by commas, for the follwing superhero {input}."
  )

  const parser = new CommaSeparatedListOutputParser()

  const chain = prompt.pipe(llm).pipe(parser)

  return await chain.invoke({
    input: "thor",
  })
}

// structure output parser
async function CallStructuredParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Extract info from the following phrase. \n{format_instruction}\n{phrase}"
  )

  const parser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "name of the person",
    age: "age of person",
    sport: "sport the person plays",
  })

  const chain = prompt.pipe(llm).pipe(parser)

  return await chain.invoke({
    phrase: "Ayush age is 69 years old. He loves to play cricket.",
    format_instruction: parser.getFormatInstructions(),
  })
}

//Zod structured output parser
async function ZodOutParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Extract info from the following phrase \n{format_instruction}\n{phrase}"
  )

  const parser = StructuredOutputParser.fromZodSchema(
    z.object({
      recipe: z.string().describe("name of the recipe"),
      ingredients: z.array(z.string()).describe("ingredients"),
    })
  )

  const chain = prompt.pipe(llm).pipe(parser)

  return await chain.invoke({
    phrase:
      "The ingredients for a Spaghetti Bolognese recipe are tomatoes, minced beef, garlic, wine and herbs.",

    format_instruction: parser.getFormatInstructions(),
  })
}

// const response = await StringParse()
// const response = await ListParser()
// const response = await CallStructuredParser()
const response = await ZodOutParser()

console.log(response)
