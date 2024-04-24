import { generateLogoutURL } from "@cloudflare/pages-plugin-cloudflare-access/api";

export const onRequest = () =>
  new Response(null, {
    status: 302,
    headers: {
      Location: generateLogoutURL({
        domain: "https://alexowusu.cloudflareaccess.com",
      }),
    },
  });