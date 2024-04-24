import cloudflareAccessPlugin from "@cloudflare/pages-plugin-cloudflare-access";

export const onRequest = cloudflareAccessPlugin({
  domain: "https://alexowusu.cloudflareaccess.com",
  aud: "113c120433356b7289d04be323da60eb23e05d38f48b0f512a4f40b72fc12370",
});