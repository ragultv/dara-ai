import type { ApiErrorBody } from "./types.ts";

export class ApiError extends Error {
  readonly status: number;
  readonly errorType: string;
  readonly code: string | undefined;
  readonly download: { model: string; [k: string]: unknown } | undefined;

  constructor(
    status: number,
    message: string,
    errorType = "api_error",
    code?: string,
    download?: { model: string; [k: string]: unknown }
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.errorType = errorType;
    this.code = code;
    this.download = download;
  }

  static fromResponseBody(
    status: number,
    body: unknown,
    defaultMessage = "Unknown API error"
  ): ApiError {
    if (typeof body === "object" && body !== null && "error" in body) {
      const errorObj = (body as ApiErrorBody).error;
      if (typeof errorObj === "object" && errorObj !== null) {
        return new ApiError(
          status,
          errorObj.message || defaultMessage,
          errorObj.type || "api_error",
          errorObj.code,
          errorObj.download
        );
      }
    }
    return new ApiError(status, typeof body === "string" ? body : defaultMessage);
  }
}
