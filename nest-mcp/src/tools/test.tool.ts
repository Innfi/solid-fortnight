import { Injectable } from "@nestjs/common";
import { type Context, Tool } from "@rekog/mcp-nest";
import z from "zod";

export interface ToolPayload {
  name: string;
}

@Injectable()
export class TestTool {
  constructor(/**private readonly repo: SomeRepository */) {}

  @Tool({
    name: "test-tool",
    description: "A test tool for demonstration purposes",
    parameters: z.object({
      name: z.string()
    }),
    // unlike mastra @tool annotation does not need output schema
  })
  public async run(payload: ToolPayload, context: Context): Promise<string> {
    context.log.info(`TestTool invoked with name: ${payload.name}`);

    return `Hello, ${payload.name}! This is a response from the test tool.`;
  }
}