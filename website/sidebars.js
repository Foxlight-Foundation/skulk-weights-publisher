module.exports = {
  tutorialSidebar: [
    "index",
    "quickstart",
    {
      type: "category",
      label: "Concepts",
      items: [
        "concepts/vindexes",
        "concepts/catalogue",
        "concepts/publishing-lifecycle",
        "concepts/safety-model",
      ],
    },
    {
      type: "category",
      label: "Guides",
      items: [
        "guides/install",
        "guides/add-catalogue-entry",
        "guides/first-publish",
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
