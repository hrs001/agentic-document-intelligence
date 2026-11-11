export function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return "session-123";

  const w = window as Window & { __ukcSessionId?: string };
  const existing = w.__ukcSessionId;
  if (existing && existing.trim()) return existing;

  const newId =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `sess_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  w.__ukcSessionId = newId;
  return newId;
}
