#!/usr/bin/env python3
"""
Interfaz gráfica para whisper.cpp
Transcripción de audio con soporte para español
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import os
import sys
import shutil
import tempfile
from pathlib import Path

# ── Rutas del proyecto ─────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.absolute()
MODELS_DIR  = SCRIPT_DIR / "models"

# Posibles ubicaciones del binario whisper-cli
_CLI_CANDIDATES = [
    SCRIPT_DIR / "build" / "bin" / "Release" / "whisper-cli.exe",
    SCRIPT_DIR / "build" / "bin" / "whisper-cli.exe",
    SCRIPT_DIR / "build" / "bin" / "Debug"   / "whisper-cli.exe",
    SCRIPT_DIR / "build" / "bin" / "whisper-cli",
    Path("whisper-cli.exe"),
    Path("whisper-cli"),
]

def _find_whisper_cli():
    for p in _CLI_CANDIDATES:
        if p.exists():
            return str(p)
    return shutil.which("whisper-cli")

def _find_ffmpeg():
    """Busca ffmpeg en PATH y en la ruta típica de instalación winget."""
    hit = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if hit:
        return hit
    # Ruta winget (se instala aquí aunque el PATH no esté actualizado en la sesión)
    winget_base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_base.exists():
        for pkg in winget_base.glob("Gyan.FFmpeg*"):
            candidates = list(pkg.rglob("ffmpeg.exe"))
            if candidates:
                return str(candidates[0])
    # Otros lugares comunes
    for p in [
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        Path(os.environ.get("ProgramFiles", "")) / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]:
        if p.exists():
            return str(p)
    return None

def _available_models():
    """Devuelve lista de (nombre, ruta) para modelos ggml presentes en models/."""
    found = []
    if MODELS_DIR.exists():
        for f in sorted(MODELS_DIR.glob("ggml-*.bin")):
            stem = f.stem.replace("ggml-", "")
            # Excluir modelos de prueba y variantes solo-inglés
            if stem.startswith("for-tests"):
                continue
            if stem.endswith((".en", ".en-q5_1", ".en-q8_0", ".en-tdrz")):
                continue
            found.append((stem, str(f)))
    return found

# Modelos multilingüe recomendados para español
RECOMMENDED_MODELS = [
    ("tiny",             "~75 MB  — Muy rápido, calidad básica"),
    ("base",             "~142 MB — Rápido, buena calidad"),
    ("small",            "~466 MB — Recomendado para español ★"),
    ("medium",           "~1.5 GB — Alta calidad"),
    ("large-v3-turbo",   "~809 MB — Excelente calidad/velocidad ★"),
    ("large-v3",         "~2.9 GB — Máxima calidad"),
]

# ── Paleta Catppuccin Mocha ────────────────────────────────────────────────────
C = {
    "bg":      "#1e1e2e",
    "bg2":     "#181825",
    "bg3":     "#313244",
    "fg":      "#cdd6f4",
    "subtext": "#a6adc8",
    "overlay": "#6c7086",
    "blue":    "#89b4fa",
    "green":   "#a6e3a1",
    "red":     "#f38ba8",
    "yellow":  "#f9e2af",
    "mauve":   "#cba6f7",
    "teal":    "#94e2d5",
    "dark":    "#11111b",
}


class WhisperGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Whisper.cpp — Transcripción en Español")
        self.root.geometry("860x780")
        self.root.minsize(700, 600)
        self.root.configure(bg=C["bg"])

        self.whisper_cli       = _find_whisper_cli()
        self._process: subprocess.Popen | None = None
        self._worker:  threading.Thread  | None = None

        self._build_styles()
        self._build_ui()
        self._refresh_models()

    # ── Estilos ttk ───────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TCombobox",
                     fieldbackground=C["bg3"], background=C["bg3"],
                     foreground=C["fg"], arrowcolor=C["blue"],
                     selectbackground=C["bg3"], selectforeground=C["fg"],
                     bordercolor=C["overlay"], lightcolor=C["bg3"],
                     darkcolor=C["bg3"])
        s.map("TCombobox",
              fieldbackground=[("readonly", C["bg3"])],
              background=[("readonly", C["bg3"])],
              foreground=[("readonly", C["fg"])])
        s.configure("TScale",
                     troughcolor=C["bg3"], background=C["blue"],
                     bordercolor=C["bg3"])
        s.configure("TProgressbar",
                     troughcolor=C["bg2"], background=C["blue"],
                     bordercolor=C["bg2"])
        s.configure("TSeparator", background=C["bg3"])

    # ── Construcción de la UI ──────────────────────────────────────────────────
    def _build_ui(self):
        # ─ Cabecera ─────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=C["bg2"], padx=20, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Whisper.cpp",
                 font=("Segoe UI", 17, "bold"), bg=C["bg2"], fg=C["blue"]
                 ).pack(side="left")
        tk.Label(hdr, text="  Transcripción automática de audio",
                 font=("Segoe UI", 10), bg=C["bg2"], fg=C["overlay"]
                 ).pack(side="left", pady=(5, 0))

        # ─ Aviso estado binary ──────────────────────────────────────────────
        banner = tk.Frame(self.root, bg=C["bg2"], padx=20, pady=4)
        banner.pack(fill="x")
        if self.whisper_cli:
            txt = f"✓  whisper-cli: {self.whisper_cli}"
            col = C["green"]
        else:
            txt = ("✗  whisper-cli no encontrado.  "
                   "Compila primero:  cmake -B build  &&  "
                   "cmake --build build -j --config Release")
            col = C["red"]
        tk.Label(banner, text=txt, font=("Segoe UI", 8),
                 bg=C["bg2"], fg=col, anchor="w", wraplength=800
                 ).pack(fill="x")

        # ─ Panel central ────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg=C["bg"], padx=18, pady=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        # ── 1. Archivo de audio ──────────────────────────────────────────────
        self._section(main, "Archivo de Audio")

        audio_row = tk.Frame(main, bg=C["bg"])
        audio_row.pack(fill="x", pady=(4, 10))
        audio_row.columnconfigure(0, weight=1)

        self.audio_var = tk.StringVar()
        ent = tk.Entry(audio_row, textvariable=self.audio_var,
                       font=("Segoe UI", 10), bg=C["bg3"], fg=C["fg"],
                       insertbackground=C["fg"], relief="flat",
                       highlightthickness=1, highlightbackground=C["overlay"],
                       highlightcolor=C["blue"])
        ent.pack(side="left", fill="x", expand=True, ipady=6)
        self._btn(audio_row, "Explorar…", self._browse_audio, C["blue"], padx=(8, 0))

        # ── 2. Modelo ────────────────────────────────────────────────────────
        self._section(main, "Modelo de IA")

        model_row = tk.Frame(main, bg=C["bg"])
        model_row.pack(fill="x", pady=(4, 2))

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var,
                                        font=("Segoe UI", 10),
                                        state="readonly", width=38)
        self.model_combo.pack(side="left", ipady=4)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_select)

        self._btn(model_row, "↻", self._refresh_models, C["overlay"], padx=(8, 4))
        self._btn(model_row, "⬇ Descargar modelo…", self._open_download_dialog, C["yellow"])

        self.model_hint = tk.Label(main, text="", font=("Segoe UI", 8),
                                   bg=C["bg"], fg=C["overlay"], anchor="w")
        self.model_hint.pack(fill="x", pady=(0, 8))

        # ── 3. Parámetros ────────────────────────────────────────────────────
        self._section(main, "Parámetros de Transcripción")

        grid = tk.Frame(main, bg=C["bg"])
        grid.pack(fill="x", pady=(6, 8))

        # Temperatura
        self._param_row(grid, 0, "Temperatura:",
                        "(0 = determinístico · 1 = más creativo)")
        self.temp_var  = tk.DoubleVar(value=0.0)
        self.temp_lbl  = self._slider_row(grid, 0, self.temp_var, 0.0, 1.0,
                                          self._fmt_temp)

        # No-speech threshold
        self._param_row(grid, 1, "Umbral silencio (no-speech):",
                        "(alto = detecta más silencios como no-habla)")
        self.nst_var   = tk.DoubleVar(value=0.6)
        self.nst_lbl   = self._slider_row(grid, 1, self.nst_var, 0.0, 1.0,
                                          self._fmt_nst)

        # Idioma
        self._param_row(grid, 2, "Idioma:")
        lang_cell = tk.Frame(grid, bg=C["bg"])
        lang_cell.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=4)

        self.lang_var = tk.StringVar(value="es")
        lang_values   = ["es", "en", "fr", "de", "pt", "it", "ja", "zh", "auto"]
        self.lang_cb  = ttk.Combobox(lang_cell, textvariable=self.lang_var,
                                      values=lang_values,
                                      font=("Segoe UI", 10),
                                      state="readonly", width=8)
        self.lang_cb.pack(side="left", ipady=4)
        self.lang_cb.bind("<<ComboboxSelected>>", self._on_lang_select)

        self._lang_names = {
            "es": "Español", "en": "English", "fr": "Français",
            "de": "Deutsch", "pt": "Português", "it": "Italiano",
            "ja": "日本語", "zh": "中文", "auto": "Detección automática"
        }
        self.lang_name_lbl = tk.Label(lang_cell, text="Español",
                                       font=("Segoe UI", 9),
                                       bg=C["bg"], fg=C["subtext"])
        self.lang_name_lbl.pack(side="left", padx=(8, 0))

        # Opciones booleanas
        self._param_row(grid, 3, "Opciones:")
        opts = tk.Frame(grid, bg=C["bg"])
        opts.grid(row=3, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=4)

        self.timestamps_var = tk.BooleanVar(value=False)
        self.translate_var  = tk.BooleanVar(value=False)

        for var, label in [(self.timestamps_var, "Incluir marcas de tiempo"),
                           (self.translate_var,  "Traducir al inglés")]:
            tk.Checkbutton(opts, text=label, variable=var,
                           font=("Segoe UI", 10),
                           bg=C["bg"], fg=C["fg"],
                           activebackground=C["bg"], activeforeground=C["blue"],
                           selectcolor=C["bg3"],
                           cursor="hand2").pack(side="left", padx=(0, 18))

        # ── 4. Botones de acción ─────────────────────────────────────────────
        act_row = tk.Frame(main, bg=C["bg"])
        act_row.pack(fill="x", pady=(4, 0))

        self.run_btn = tk.Button(
            act_row, text="▶  Transcribir",
            font=("Segoe UI", 12, "bold"),
            bg=C["blue"], fg=C["dark"],
            activebackground="#7aa2f7", activeforeground=C["dark"],
            relief="flat", cursor="hand2", padx=24, pady=8,
            command=self._start_transcription)
        self.run_btn.pack(side="left")

        self.stop_btn = tk.Button(
            act_row, text="■  Detener",
            font=("Segoe UI", 12, "bold"),
            bg=C["red"], fg=C["dark"],
            activebackground="#eb6f92", activeforeground=C["dark"],
            relief="flat", cursor="hand2", padx=24, pady=8,
            state="disabled",
            command=self._stop_transcription)
        self.stop_btn.pack(side="left", padx=(8, 0))

        self._btn(act_row, "📋 Copiar", self._copy_output, C["overlay"],
                  padx=(0, 4), side="right")
        self._btn(act_row, "🗑 Limpiar", self._clear_output, C["overlay"],
                  side="right")

        # ── 5. Área de salida ────────────────────────────────────────────────
        self._section(main, "Transcripción")

        txt_frame = tk.Frame(main, bg=C["bg3"],
                             highlightbackground=C["overlay"],
                             highlightthickness=1)
        txt_frame.pack(fill="both", expand=True, pady=(4, 0))

        self.out_text = scrolledtext.ScrolledText(
            txt_frame, font=("Consolas", 10),
            bg=C["bg3"], fg=C["fg"],
            insertbackground=C["fg"],
            relief="flat", padx=12, pady=10,
            wrap="word", state="disabled")
        self.out_text.pack(fill="both", expand=True)

        # ── Barra de estado ──────────────────────────────────────────────────
        sbar = tk.Frame(self.root, bg=C["bg2"], padx=18, pady=5)
        sbar.pack(fill="x", side="bottom")

        self.status_var = tk.StringVar(value="Listo.")
        self.status_lbl = tk.Label(sbar, textvariable=self.status_var,
                                   font=("Segoe UI", 9),
                                   bg=C["bg2"], fg=C["overlay"], anchor="w")
        self.status_lbl.pack(side="left", fill="x", expand=True)

        self.progress = ttk.Progressbar(sbar, mode="indeterminate", length=120)
        self.progress.pack(side="right")

    # ── Helpers de construcción ────────────────────────────────────────────────
    def _section(self, parent, title: str):
        f = tk.Frame(parent, bg=C["bg"])
        f.pack(fill="x", pady=(10, 0))
        tk.Label(f, text=title, font=("Segoe UI", 10, "bold"),
                 bg=C["bg"], fg=C["mauve"]).pack(side="left")
        tk.Frame(f, bg=C["bg3"], height=1).pack(
            side="left", fill="x", expand=True, padx=(8, 0), pady=(2, 0))

    def _btn(self, parent, text, cmd, color, padx=(0, 0), side="left"):
        b = tk.Button(parent, text=text, command=cmd,
                      font=("Segoe UI", 9), bg=color, fg=C["dark"],
                      activebackground=color, relief="flat",
                      cursor="hand2", padx=10, pady=6)
        b.pack(side=side, padx=padx)
        return b

    def _param_row(self, grid, row, label, hint=""):
        tk.Label(grid, text=label, font=("Segoe UI", 10, "bold"),
                 bg=C["bg"], fg=C["fg"], width=30, anchor="w"
                 ).grid(row=row, column=0, sticky="w", pady=4)
        if hint:
            tk.Label(grid, text=hint, font=("Segoe UI", 8),
                     bg=C["bg"], fg=C["overlay"]
                     ).grid(row=row, column=2, sticky="w", padx=12)

    def _slider_row(self, grid, row, var, from_, to, fmt_cb):
        cell = tk.Frame(grid, bg=C["bg"])
        cell.grid(row=row, column=1, sticky="w", padx=(8, 0))
        val_lbl = tk.Label(cell, text=f"{var.get():.2f}",
                           font=("Consolas", 10, "bold"),
                           bg=C["bg"], fg=C["blue"], width=5)
        val_lbl.pack(side="right", padx=(8, 0))
        ttk.Scale(cell, from_=from_, to=to, variable=var,
                  orient="horizontal", length=300,
                  command=lambda v, lbl=val_lbl, cb=fmt_cb: lbl.config(text=cb(v))
                  ).pack(side="left")
        return val_lbl

    @staticmethod
    def _fmt_temp(v): return f"{float(v):.2f}"
    @staticmethod
    def _fmt_nst(v):  return f"{float(v):.2f}"

    # ── Handlers de eventos ───────────────────────────────────────────────────
    def _browse_audio(self):
        path = filedialog.askopenfilename(
            title="Seleccionar archivo de audio",
            filetypes=[
                ("Audio / Video",
                 "*.wav *.mp3 *.m4a *.ogg *.flac *.aac *.wma *.opus *.mp4 *.mkv *.webm"),
                ("Todos los archivos", "*.*"),
            ])
        if path:
            self.audio_var.set(path)
            self._set_status(f"Archivo: {Path(path).name}")

    def _refresh_models(self):
        models = _available_models()
        names  = [m[0] for m in models]
        self.model_combo["values"] = names
        if names:
            preferred = ["small", "base", "large-v3-turbo",
                         "medium", "large-v3", "tiny"]
            sel = next((p for p in preferred if p in names), names[0])
            self.model_var.set(sel)
            self._on_model_select()
            self._set_status(f"{len(names)} modelo(s) encontrado(s) en ./models/")
        else:
            self.model_var.set("")
            self.model_hint.config(text="")
            self._set_status(
                "Sin modelos. Usa '⬇ Descargar modelo…' para obtener uno.",
                C["yellow"])

    def _on_model_select(self, _=None):
        name = self.model_var.get()
        for m, desc in RECOMMENDED_MODELS:
            if name == m:
                self.model_hint.config(text=f"   {desc}")
                return
        self.model_hint.config(text="")

    def _on_lang_select(self, _=None):
        self.lang_name_lbl.config(
            text=self._lang_names.get(self.lang_var.get(), ""))

    # ── Descarga de modelos ───────────────────────────────────────────────────
    def _open_download_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Descargar modelo")
        dlg.geometry("520x400")
        dlg.configure(bg=C["bg2"])
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        tk.Label(dlg, text="Selecciona un modelo multilingüe",
                 font=("Segoe UI", 12, "bold"),
                 bg=C["bg2"], fg=C["blue"]).pack(pady=(18, 4), padx=18)
        tk.Label(dlg,
                 text="Para español usa modelos sin '.en'.\n"
                      "Se recomiendan 'small' (equilibrio) o 'large-v3-turbo' (calidad).",
                 font=("Segoe UI", 9), bg=C["bg2"], fg=C["subtext"],
                 justify="left", wraplength=480).pack(padx=18, anchor="w")

        sel = tk.StringVar(value="small")
        for name, desc in RECOMMENDED_MODELS:
            row = tk.Frame(dlg, bg=C["bg2"])
            row.pack(fill="x", padx=30, pady=1)
            tk.Radiobutton(row, text=f"{name:<20}  {desc}",
                           variable=sel, value=name,
                           font=("Consolas", 9),
                           bg=C["bg2"], fg=C["fg"],
                           activebackground=C["bg2"],
                           selectcolor=C["bg3"],
                           cursor="hand2").pack(anchor="w")

        btn_row = tk.Frame(dlg, bg=C["bg2"])
        btn_row.pack(pady=18)

        def do_dl():
            model = sel.get()
            dlg.destroy()
            self._run_download(model)

        tk.Button(btn_row, text="⬇ Descargar", command=do_dl,
                  font=("Segoe UI", 10, "bold"),
                  bg=C["blue"], fg=C["dark"], relief="flat",
                  padx=20, pady=8, cursor="hand2").pack(side="left", padx=8)
        tk.Button(btn_row, text="Cancelar", command=dlg.destroy,
                  font=("Segoe UI", 10),
                  bg=C["overlay"], fg=C["dark"], relief="flat",
                  padx=20, pady=8, cursor="hand2").pack(side="left", padx=8)

    def _run_download(self, model: str):
        script = MODELS_DIR / "download-ggml-model.cmd"
        if not script.exists():
            messagebox.showerror("Error", f"No se encontró: {script}")
            return
        self._set_status(f"Descargando '{model}'…", C["yellow"])
        self.progress.start(10)

        def worker():
            try:
                r = subprocess.run(
                    ["cmd", "/c", str(script), model],
                    capture_output=True, text=True, cwd=str(SCRIPT_DIR))
                if r.returncode == 0:
                    self.root.after(0, lambda: self._set_status(
                        f"✓ Modelo '{model}' descargado.", C["green"]))
                    self.root.after(0, self._refresh_models)
                else:
                    err = (r.stderr or r.stdout)[:200]
                    self.root.after(0, lambda: self._set_status(
                        f"Error al descargar: {err}", C["red"]))
            except Exception as exc:
                self.root.after(0, lambda: self._set_status(
                    f"Error: {exc}", C["red"]))
            finally:
                self.root.after(0, self.progress.stop)

        threading.Thread(target=worker, daemon=True).start()

    # ── Transcripción ─────────────────────────────────────────────────────────
    def _start_transcription(self):
        if not self.whisper_cli:
            messagebox.showerror(
                "Binario no encontrado",
                "No se encontró whisper-cli.\n\n"
                "Compila el proyecto:\n"
                "  cmake -B build\n"
                "  cmake --build build -j --config Release")
            return

        audio = self.audio_var.get().strip()
        if not audio:
            messagebox.showwarning("Sin archivo", "Selecciona un archivo de audio.")
            return
        if not os.path.exists(audio):
            messagebox.showerror("Error", f"Archivo no encontrado:\n{audio}")
            return

        model_name = self.model_var.get()
        if not model_name:
            messagebox.showwarning("Sin modelo",
                                   "Selecciona o descarga un modelo primero.")
            return
        model_path = MODELS_DIR / f"ggml-{model_name}.bin"
        if not model_path.exists():
            messagebox.showerror(
                "Modelo no encontrado",
                f"No se encontró:\n{model_path}\n\n"
                "Usa '⬇ Descargar modelo…' para obtenerlo.")
            return

        self._write_out("", clear=True)
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._set_status("Procesando…", C["yellow"])
        self.progress.start(10)

        self._worker = threading.Thread(
            target=self._transcribe_worker,
            args=(audio, str(model_path)),
            daemon=True)
        self._worker.start()

    def _stop_transcription(self):
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass
        self._set_status("Deteniendo…", C["yellow"])

    def _transcribe_worker(self, audio_path: str, model_path: str):
        # Formatos soportados de forma nativa por whisper-cli
        NATIVE_FORMATS = {".wav", ".mp3", ".ogg", ".flac"}
        tmp_wav = None
        try:
            wav_path = audio_path
            ext = Path(audio_path).suffix.lower()

            # Convertir a WAV 16 kHz mono solo si el formato no es nativo
            if ext not in NATIVE_FORMATS:
                ffmpeg_bin = _find_ffmpeg()
                if not ffmpeg_bin:
                    self.root.after(0, lambda: self._write_out(
                        f"⚠  El formato '{ext}' no es compatible directamente.\n"
                        "  Formatos nativos: wav, mp3, ogg, flac\n\n"
                        "  Para otros formatos instala ffmpeg:\n"
                        "    winget install Gyan.FFmpeg\n"
                        "  O convierte manualmente:\n"
                        "    ffmpeg -i entrada.m4a -ar 16000 -ac 1 -c:a pcm_s16le salida.wav\n"))
                    self.root.after(0, lambda: self._set_status(
                        f"Formato '{ext}' no soportado sin ffmpeg.", C["red"]))
                    return
                self.root.after(0, lambda: self._set_status(
                    "Convirtiendo audio con ffmpeg…", C["yellow"]))
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                tmp_wav = tmp.name
                conv = subprocess.run(
                    [ffmpeg_bin, "-y", "-i", audio_path,
                     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", tmp_wav],
                    capture_output=True, text=True)
                if conv.returncode != 0:
                    err = conv.stderr[-600:]
                    self.root.after(0, lambda: self._write_out(
                        f"Error en conversión ffmpeg:\n{err}\n"))
                    self.root.after(0, lambda: self._set_status(
                        "Error en conversión.", C["red"]))
                    return
                wav_path = tmp_wav

            # Construir comando whisper-cli
            cmd = [
                self.whisper_cli,
                "-m",   model_path,
                "-f",   wav_path,
                "-l",   self.lang_var.get(),
                "-tp",  f"{self.temp_var.get():.3f}",
                "-nth", f"{self.nst_var.get():.3f}",
            ]
            if not self.timestamps_var.get():
                cmd.append("-nt")
            if self.translate_var.get():
                cmd.append("-tr")

            self.root.after(0, lambda: self._set_status(
                "Transcribiendo…", C["yellow"]))

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            # Leer stdout en tiempo real
            for line in self._process.stdout:
                self.root.after(0, lambda l=line: self._write_out(l))

            stderr_out = self._process.stderr.read()
            self._process.wait()
            rc = self._process.returncode

            if rc == 0:
                self.root.after(0, lambda: self._set_status(
                    "✓ Transcripción completada.", C["green"]))
            else:
                preview = (stderr_out or "")[:400]
                if preview:
                    self.root.after(0, lambda p=preview: self._write_out(
                        f"\n── Detalles de error ──\n{p}\n"))
                self.root.after(0, lambda: self._set_status(
                    f"Error (código {rc}).", C["red"]))

        except Exception as exc:
            self.root.after(0, lambda: self._write_out(f"\nError inesperado: {exc}\n"))
            self.root.after(0, lambda: self._set_status(f"Error: {exc}", C["red"]))
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    pass
            self._process = None
            self.root.after(0, self._on_done)

    def _on_done(self):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    # ── Helpers de salida ─────────────────────────────────────────────────────
    def _write_out(self, text: str, clear: bool = False):
        self.out_text.config(state="normal")
        if clear:
            self.out_text.delete("1.0", "end")
        if text:
            self.out_text.insert("end", text)
            self.out_text.see("end")
        self.out_text.config(state="disabled")

    def _copy_output(self):
        txt = self.out_text.get("1.0", "end").strip()
        if txt:
            self.root.clipboard_clear()
            self.root.clipboard_append(txt)
            self._set_status("Texto copiado al portapapeles.", C["teal"])
        else:
            self._set_status("Nada que copiar.", C["overlay"])

    def _clear_output(self):
        self._write_out("", clear=True)
        self._set_status("Listo.")

    def _set_status(self, msg: str, color: str = C["overlay"]):
        self.status_var.set(msg)
        self.status_lbl.config(fg=color)


def main():
    root = tk.Tk()
    try:
        root.iconbitmap(default="")
    except Exception:
        pass
    WhisperGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()