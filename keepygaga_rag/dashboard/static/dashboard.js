(() => {
  if (document.body.dataset.dashboardAutoClose === "true") {
    let presenceSource = null;
    let pageIsLeaving = false;

    const connectPresence = () => {
      if (pageIsLeaving || presenceSource !== null) return;
      presenceSource = new EventSource("/_dashboard/presence");
    };

    window.addEventListener("pagehide", () => {
      pageIsLeaving = true;
      presenceSource?.close();
      presenceSource = null;
    });
    window.addEventListener("pageshow", () => {
      pageIsLeaving = false;
      connectPresence();
    });
    connectPresence();
  }

  const validating = document.documentElement.lang === "zh-CN"
    ? "正在校验…"
    : "Validating…";
  const chooseButton = document.querySelector("[data-choose-directory]");
  const directoryPath = document.querySelector("[data-directory-path]");
  const csrfToken = document.querySelector(
    '.source-probe-form input[name="csrf_token"]',
  );
  if (chooseButton && directoryPath && csrfToken) {
    const idleLabel = chooseButton.textContent;
    chooseButton.addEventListener("click", async () => {
      chooseButton.disabled = true;
      chooseButton.textContent = document.documentElement.lang === "zh-CN"
        ? "等待选择…"
        : "Waiting…";
      const payload = new FormData();
      payload.set("csrf_token", csrfToken.value);
      try {
        const response = await fetch("/knowledge/choose-directory", {
          method: "POST",
          body: payload,
        });
        const result = await response.json();
        if (!response.ok || result.status === "error") {
          window.alert(result.message || "Unable to choose a directory.");
        } else if (result.status === "ok") {
          directoryPath.value = result.path;
          directoryPath.dispatchEvent(new Event("input", { bubbles: true }));
        }
      } catch {
        window.alert(
          document.documentElement.lang === "zh-CN"
            ? "无法打开系统目录选择器。"
            : "Unable to open the system folder picker.",
        );
      } finally {
        chooseButton.disabled = false;
        chooseButton.textContent = idleLabel;
      }
    });
  }

  const withinScope = (path, directory) => (
    directory === "."
    || path === directory
    || path.startsWith(`${directory}/`)
  );

  document.querySelectorAll("[data-scope-picker]").forEach((picker) => {
    const nodes = [...picker.querySelectorAll("[data-scope-node]")];
    const rows = [...picker.querySelectorAll("[data-scope-row]")];
    const count = picker.querySelector("[data-scope-count]");
    const hidden = picker.querySelector("[data-scope-selection]");
    const filter = picker.querySelector("[data-scope-filter]");
    const form = picker.closest("form");
    const directories = nodes
      .filter((node) => node.dataset.scopeKind === "directory")
      .sort((left, right) => (
        right.dataset.scopePath.split("/").length
        - left.dataset.scopePath.split("/").length
      ));
    const files = nodes.filter((node) => node.dataset.scopeKind === "file");
    const expandedDirectories = new Set();

    const refreshVisibility = () => {
      const query = filter?.value.trim().toLocaleLowerCase() || "";
      rows.forEach((row) => {
        if (query) {
          row.hidden = !row.dataset.scopeSearch.includes(query);
          return;
        }
        row.hidden = directories.some((directory) => (
          directory.dataset.scopePath !== row.dataset.scopePath
          && withinScope(row.dataset.scopePath, directory.dataset.scopePath)
          && !expandedDirectories.has(directory.dataset.scopePath)
        ));
      });
    };

    picker.querySelectorAll("[data-scope-toggle]").forEach((toggle) => {
      const row = toggle.closest("[data-scope-row]");
      const path = row.dataset.scopePath;
      const icon = toggle.querySelector("[data-scope-toggle-icon]");
      const toggleDirectory = () => {
        const expanded = !expandedDirectories.has(path);
        if (expanded) {
          expandedDirectories.add(path);
        } else {
          expandedDirectories.delete(path);
        }
        toggle.setAttribute("aria-expanded", String(expanded));
        toggle.setAttribute(
          "aria-label",
          document.documentElement.lang === "zh-CN"
            ? (expanded ? "收起目录" : "展开目录")
            : (expanded ? "Collapse directory" : "Expand directory"),
        );
        if (icon) icon.textContent = expanded ? "▾" : "▸";
        refreshVisibility();
      };
      toggle.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        toggleDirectory();
      });
    });

    const refreshDirectories = () => {
      directories.forEach((directory) => {
        const path = directory.dataset.scopePath;
        const descendants = files.filter((file) => (
          withinScope(file.dataset.scopePath, path)
        ));
        const selected = descendants.filter((file) => file.checked).length;
        directory.checked = descendants.length > 0
          && selected === descendants.length;
        directory.indeterminate = selected > 0 && selected < descendants.length;
      });
      if (count) {
        count.textContent = String(files.filter((file) => file.checked).length);
      }
    };

    nodes.forEach((node) => {
      node.indeterminate = node.dataset.scopeState === "partial";
      node.addEventListener("change", () => {
        if (node.dataset.scopeKind === "directory") {
          const path = node.dataset.scopePath;
          nodes.forEach((candidate) => {
            if (
              candidate !== node
              && withinScope(candidate.dataset.scopePath, path)
            ) {
              candidate.checked = node.checked;
              candidate.indeterminate = false;
            }
          });
        }
        refreshDirectories();
      });
    });

    picker.querySelector("[data-scope-all]")?.addEventListener("click", () => {
      nodes.forEach((node) => {
        node.checked = true;
        node.indeterminate = false;
      });
      refreshDirectories();
    });
    picker.querySelector("[data-scope-none]")?.addEventListener("click", () => {
      nodes.forEach((node) => {
        node.checked = false;
        node.indeterminate = false;
      });
      refreshDirectories();
    });
    filter?.addEventListener("input", () => {
      refreshVisibility();
    });
    form?.addEventListener("submit", () => {
      const checked = nodes.filter((node) => node.checked);
      const selection = checked
        .filter((node) => !checked.some((candidate) => (
          candidate !== node
          && candidate.dataset.scopeKind === "directory"
          && withinScope(
            node.dataset.scopePath,
            candidate.dataset.scopePath,
          )
        )))
        .map((node) => node.dataset.scopePath);
      hidden.value = JSON.stringify(selection);
    });
    refreshDirectories();
    refreshVisibility();
  });

  document.querySelectorAll("[data-submit-lock]").forEach((form) => {
    form.addEventListener("submit", () => {
      const button = form.querySelector("[data-save-button]");
      if (!button) return;
      button.disabled = true;
      button.textContent = validating;
    });
  });
})();
