import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CareCopilot",
  description:
    "An AI assistant that answers questions about a patient record, with citations.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
