{
  description = "Whisper model management, transcription CLI and Python SDK";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix.url = "github:pyproject-nix/uv2nix";
    uv2nix.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.url = "github:pyproject-nix/build-system-pkgs";
    pyproject-build-systems.inputs.nixpkgs.follows = "nixpkgs";
    pyproject-build-systems.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.inputs.uv2nix.follows = "uv2nix";
  };
  outputs = {
    self,
    nixpkgs,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
    ...
  }: let
    systems = ["x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin"];
    each-system = nixpkgs.lib.genAttrs systems;
    workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = self;};
    overlay = workspace.mkPyprojectOverlay {sourcePreference = "wheel";};
    mk-application = pkgs: (pkgs.callPackages pyproject-nix.build.util {}).mkApplication;
  in {
    lib.runtimeOverlay = import ./nix/runtime-overlay.nix;
    lib.mkPythonSet = {
      pkgs,
      python ? pkgs.python314,
    }:
      (pkgs.callPackage pyproject-nix.build.packages {inherit python;}).overrideScope (pkgs.lib.composeManyExtensions [
        pyproject-build-systems.overlays.wheel
        overlay
        (self.lib.runtimeOverlay {inherit pkgs;})
      ]);
    lib.cudaLibraryPath = pkgs: pkgs.lib.makeLibraryPath [pkgs.cudaPackages.cuda_cudart pkgs.cudaPackages.libcublas pkgs.cudaPackages.libcurand pkgs.cudaPackages.cudnn];
    lib.mkPackage = {
      pkgs,
      cudaSupport ? false,
      python ? pkgs.python314,
    }: let
      python-set = self.lib.mkPythonSet {inherit pkgs python;};
      environment = python-set.mkVirtualEnv "audio-transcribe-env" {audio-transcribe = [];};
      app = mk-application pkgs {
        venv = environment;
        package = python-set.audio-transcribe;
      };
    in
      assert !cudaSupport || (pkgs.stdenv.hostPlatform.isLinux && pkgs.stdenv.hostPlatform.isx86_64);
        app.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pkgs.installShellFiles pkgs.makeWrapper];
          postInstall = (old.postInstall or "") + "\ninstallShellCompletion --zsh ${self}/_audio-transcribe";
          postFixup =
            (old.postFixup or "")
            + pkgs.lib.optionalString cudaSupport ''
              wrapProgram "$out/bin/audio-transcribe" --prefix LD_LIBRARY_PATH : "${self.lib.cudaLibraryPath pkgs}:/run/opengl-driver/lib:/usr/lib/wsl/lib"
            '';
          meta.mainProgram = "audio-transcribe";
          passthru =
            (old.passthru or {})
            // {
              python-env = environment;
              cuda-library-path = self.lib.cudaLibraryPath pkgs;
              tests.unit = pkgs.runCommand "audio-transcribe-unit" {} ''
                export PYTHONPATH=${self}/src
                export PYTHONDONTWRITEBYTECODE=1
                ${python.interpreter} -m unittest discover -s ${self}/tests
                touch "$out"
              '';
            };
        });
    packages = each-system (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
      {
        default = self.lib.mkPackage {inherit pkgs;};
        audio-transcribe = self.packages.${system}.default;
      }
      // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
        cuda = self.lib.mkPackage {
          inherit pkgs;
          cudaSupport = true;
        };
      });
    checks = each-system (system: {
      unit = self.packages.${system}.default.tests.unit;
      imports = nixpkgs.legacyPackages.${system}.runCommand "audio-transcribe-imports" {} ''
        ${self.packages.${system}.default.python-env}/bin/python - <<'PYCODE'
        import tempfile
        import wave
        from pathlib import Path
        import audio_transcribe, faster_whisper, ctranslate2, av
        from faster_whisper.audio import decode_audio
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(bytes(3200))
            assert len(decode_audio(str(path))) == 1600
        PYCODE
        touch "$out"
      '';
    });
    devShells = each-system (system: let pkgs = nixpkgs.legacyPackages.${system}; in {default = pkgs.mkShell {packages = [pkgs.uv pkgs.python314];};});
    formatter = each-system (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
