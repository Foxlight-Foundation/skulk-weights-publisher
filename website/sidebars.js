module.exports = {
  tutorialSidebar: [
    "index",
    "quickstart",
    {
      type: "category",
      label: "Concepts",
      items: [
        "concepts/how-skulk-works",
        "concepts/vindexes",
        "concepts/catalog",
        "concepts/publishing-lifecycle",
        "concepts/safety-model",
        "concepts/assistant-models",
      ],
    },
    {
      type: "category",
      label: "Guides",
      items: [
        "guides/install",
        "guides/add-catalog-entry",
        "guides/first-publish",
        "guides/mtp-sidecar",
        "guides/vision-sidecar",
        "guides/gemma4-assistant",
        "guides/skulk-ui",
        "guides/runner-setup",
        "guides/development",
      ],
    },
    {
      type: "category",
      label: "Reference",
      items: [
        "reference/cli",
        "reference/manifest",
        "reference/environment",
        "reference/workflows",
        "reference/errors",
      ],
    },
    "troubleshooting",
  ],
};
