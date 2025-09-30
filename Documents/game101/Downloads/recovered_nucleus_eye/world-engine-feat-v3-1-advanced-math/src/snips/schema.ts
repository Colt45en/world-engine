import Ajv, { JSONSchemaType } from "ajv";

export interface Snippet {
    id: string;
    title: string;
    tags: string[];
    lang?: string;
    source?: string;
    notes?: string;
    code: string;
    hash: string;
    createdAt: string; // ISO 8601
    updatedAt: string; // ISO 8601
}

const ajv = new Ajv({ allErrors: true, strict: true });

const schema: JSONSchemaType<Snippet> = {
    type: "object",
    additionalProperties: false,
    required: ["id", "title", "tags", "code", "hash", "createdAt", "updatedAt"],
    properties: {
        id: { type: "string", minLength: 1 },
        title: { type: "string", minLength: 1 },
        tags: { type: "array", items: { type: "string" }, default: [] },
        lang: { type: "string", nullable: true, minLength: 1 },
        source: { type: "string", nullable: true, minLength: 1 },
        notes: { type: "string", nullable: true },
        code: { type: "string", minLength: 1 },
        hash: { type: "string", minLength: 32 },
        createdAt: { type: "string", minLength: 10 },
        updatedAt: { type: "string", minLength: 10 }
    }
};

const validate = ajv.compile(schema);

export function validateSnippet(obj: unknown): Snippet {
    if (!validate(obj)) {
        const msg = ajv.errorsText(validate.errors, { separator: " | " });
        throw new Error(`Invalid Snippet: ${msg}`);
    }
    return obj as Snippet;
}
