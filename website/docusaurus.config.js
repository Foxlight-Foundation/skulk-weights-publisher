const config = {
  title: "SWP: Skulk Weights Publisher",
  tagline: "Prepare and publish weight artifacts for Skulk.",
  favicon: "img/skulk-logo.svg",

  url: "https://foxlight-foundation.github.io",
  baseUrl: process.env.DOCUSAURUS_BASE_URL || "/skulk-weights-publisher/",
  organizationName: "Foxlight-Foundation",
  projectName: "skulk-weights-publisher",

  onBrokenLinks: "throw",
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          routeBasePath: "/",
          sidebarPath: require.resolve("./sidebars.js"),
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: "SWP",
      logo: {
        alt: "Skulk Logo",
        src: "img/skulk-logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "Docs",
        },
        {
          href: "https://github.com/Foxlight-Foundation/skulk-weights-publisher",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            { label: "Quickstart", to: "/quickstart" },
            { label: "CLI reference", to: "/reference/cli" },
          ],
        },
        {
          title: "Project",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/Foxlight-Foundation/skulk-weights-publisher",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Foxlight Foundation.`,
    },
    prism: {
      additionalLanguages: ["bash", "yaml", "json"],
    },
  },
};

module.exports = config;
