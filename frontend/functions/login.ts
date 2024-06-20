import { generateLoginURL } from "@cloudflare/pages-plugin-cloudflare-access/api";

export const onRequest = () => {
  const loginURL = generateLoginURL({
    redirectURL: "https://fyp.alexo.uk",
    domain: "https://alexowusu.cloudflareaccess.com",
    aud: "113c120433356b7289d04be323da60eb23e05d38f48b0f512a4f40b72fc12370",
  });

  return new Response(null, {
    status: 302,
    headers: { Location: loginURL },
  });
};