"use client";

import { useEffect, useState } from "react";
import { getToken, logout } from "@/lib/api";
import { Chat } from "@/components/Chat";
import { Login } from "@/components/Login";

export default function Home() {
  // Undefined until the first client render: the token lives in
  // sessionStorage, which does not exist during prerender, so rendering
  // either view before that check would flash the wrong one.
  const [authed, setAuthed] = useState<boolean | undefined>(undefined);

  useEffect(() => {
    setAuthed(Boolean(getToken()));
  }, []);

  if (authed === undefined) return null;

  return authed ? (
    <Chat
      onSignOut={() => {
        logout();
        setAuthed(false);
      }}
    />
  ) : (
    <Login onAuthenticated={() => setAuthed(true)} />
  );
}
