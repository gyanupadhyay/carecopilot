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
  // Why the sign-in screen is showing, when it is showing for a reason.
  // Without it an expired session looks like the app forgot the login.
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    setAuthed(Boolean(getToken()));
  }, []);

  if (authed === undefined) return null;

  return authed ? (
    <Chat
      onSignOut={() => {
        logout();
        setNotice(null);
        setAuthed(false);
      }}
      onSessionExpired={() => {
        // The token is already cleared by the API layer; this is the part
        // it cannot do — putting the sign-in screen back on screen.
        setNotice("Your session has expired. Please sign in again.");
        setAuthed(false);
      }}
    />
  ) : (
    <Login
      notice={notice}
      onAuthenticated={() => {
        setNotice(null);
        setAuthed(true);
      }}
    />
  );
}
