import { describe, it, expect, beforeEach } from "vitest";
import { chatStore } from "../../src/lib/stores/chat.svelte.ts";

describe("ChatStore", () => {
  beforeEach(() => {
    localStorage.clear();
    chatStore.conversations = [];
    chatStore.activeConversationId = null;
  });

  it("creates and selects a new conversation", () => {
    const id = chatStore.createConversation("Test Conversation");
    expect(chatStore.conversations.length).toBe(1);
    expect(chatStore.activeConversationId).toBe(id);
    expect(chatStore.activeConversation?.title).toBe("Test Conversation");
  });

  it("renames conversation", () => {
    const id = chatStore.createConversation("Original");
    chatStore.renameConversation(id, "Updated Title");
    expect(chatStore.activeConversation?.title).toBe("Updated Title");
  });

  it("deletes conversation and updates active selection", () => {
    const id1 = chatStore.createConversation("First");
    const id2 = chatStore.createConversation("Second");

    expect(chatStore.activeConversationId).toBe(id2);
    chatStore.deleteConversation(id2);

    expect(chatStore.conversations.length).toBe(1);
    expect(chatStore.activeConversationId).toBe(id1);
  });
});
