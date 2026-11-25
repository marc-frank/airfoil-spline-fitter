// Files and folders for the browser version through the File System Access
// API (Chrome and Edge; elsewhere `supported` is false and the app falls back
// to the file picker and downloads). Used from crates/spline-fitter-gui/src/web.rs.
//
// Every file or folder the user picks gets an id; its handle is kept in
// IndexedDB under that id, so Save writes back to the opened file and the
// recent files reopen after a reload (the browser asks for permission again).
(() => {
  const DB = "airfoil-spline-fitter";
  const STORE = "handles";

  function db() {
    return new Promise((resolve, reject) => {
      const open = indexedDB.open(DB, 1);
      open.onupgradeneeded = () => open.result.createObjectStore(STORE);
      open.onsuccess = () => resolve(open.result);
      open.onerror = () => reject(open.error);
    });
  }

  async function keep(handle) {
    const id = Math.random().toString(36).slice(2, 10);
    const d = await db();
    await new Promise((resolve, reject) => {
      const tx = d.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(handle, id);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
    return id;
  }

  async function handle(id) {
    const d = await db();
    return new Promise((resolve, reject) => {
      const get = d.transaction(STORE).objectStore(STORE).get(id);
      get.onsuccess = () => resolve(get.result || null);
      get.onerror = () => reject(get.error);
    });
  }

  async function permitted(h, mode) {
    const options = { mode };
    if ((await h.queryPermission(options)) === "granted") return true;
    return (await h.requestPermission(options)) === "granted";
  }

  function types(description, extensions) {
    return [{ description, accept: { "application/octet-stream": extensions.map((e) => "." + e) } }];
  }

  async function read(id, h) {
    const file = await h.getFile();
    return { id, name: file.name, bytes: new Uint8Array(await file.arrayBuffer()) };
  }

  async function write(h, text) {
    const w = await h.createWritable();
    await w.write(text);
    await w.close();
  }

  // An aborted picker answers null; other failures reject with their message.
  const quiet = (e) => {
    if (e && e.name === "AbortError") return null;
    throw String((e && e.message) || e);
  };

  window.splineFitterFiles = {
    supported: () => typeof window.showOpenFilePicker === "function",

    // { id, name, bytes } or null.
    open: (description, extensions) =>
      window
        .showOpenFilePicker({ id: "airfoils", types: types(description, extensions) })
        .then(async ([h]) => read(await keep(h), h))
        .catch(quiet),

    // The file of a recent-files entry: { id, name, bytes } or null.
    reopen: async (id) => {
      const h = await handle(id);
      if (!h || !(await permitted(h, "read"))) return null;
      return read(id, h).catch(() => null);
    },

    // Asks for a place and writes there: { id, name } or null.
    saveAs: (suggestedName, description, extensions, text) =>
      window
        .showSaveFilePicker({ id: "airfoils", suggestedName, types: types(description, extensions) })
        .then(async (h) => {
          await write(h, text);
          return { id: await keep(h), name: h.name };
        })
        .catch(quiet),

    // Writes over a file picked before; true when written.
    save: async (id, text) => {
      const h = await handle(id);
      if (!h || !(await permitted(h, "readwrite"))) return false;
      await write(h, text);
      return true;
    },

    // { id, name } or null.
    pickFolder: () =>
      window
        .showDirectoryPicker({ id: "batch", mode: "readwrite" })
        .then(async (h) => ({ id: await keep(h), name: h.name }))
        .catch(quiet),

    // A batch: { sources: [[name, bytes], ...], existing: [name, ...] } (the
    // source folder's .dat files that are not spline files, and every name
    // in the target folder).
    readBatch: async (sourceId, targetId) => {
      const source = await handle(sourceId);
      const target = await handle(targetId);
      if (!source || !target || !(await permitted(source, "read")) || !(await permitted(target, "readwrite"))) {
        throw "The browser did not allow access to the folders";
      }
      const sources = [];
      for await (const [name, h] of source.entries()) {
        const lower = name.toLowerCase();
        if (h.kind === "file" && lower.endsWith(".dat") && !lower.includes(".spline.")) {
          sources.push([name, new Uint8Array(await (await h.getFile()).arrayBuffer())]);
        }
      }
      const existing = [];
      for await (const name of target.keys()) existing.push(name);
      return { sources, existing };
    },

    // Writes a file into a folder picked before.
    writeInFolder: async (folderId, name, text) => {
      const folder = await handle(folderId);
      if (!folder || !(await permitted(folder, "readwrite"))) throw "The browser did not allow writing to the folder";
      await write(await folder.getFileHandle(name, { create: true }), text);
    },
  };
})();
