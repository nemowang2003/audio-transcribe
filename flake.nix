{
  description = "Whisper model management, transcription CLI and Python SDK";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
  };
  outputs = {
    self,
    nixpkgs,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
  }: let
    systems = ["x86_64-linux" "aarch64-linux" "aarch64-darwin"];
    each-system = nixpkgs.lib.genAttrs systems;
    workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = self;};
    overlay = workspace.mkPyprojectOverlay {sourcePreference = "wheel";};
  in {
    lib.mkPackage = {
      pkgs,
      cudaSupport ? false,
      python ? pkgs.python314,
    }: let
      pythonSet = (pkgs.callPackage pyproject-nix.build.packages {inherit python;}).overrideScope (
        pkgs.lib.composeManyExtensions [
          pyproject-build-systems.overlays.wheel
          overlay
          (final: prev: {
            audio-transcribe = prev.audio-transcribe.overrideAttrs (old: {
              # Use the locked backend instead of the version bundled with uv.
              uvBuildFlags = ["--force-pep517"];
              nativeBuildInputs = (old.nativeBuildInputs or []) ++ final.resolveBuildSystem {uv-build = [];};
            });
          })
        ]
      );
      virtualenv = pythonSet.mkVirtualEnv "audio-transcribe-env" workspace.deps.default;
      application = (pkgs.callPackage pyproject-nix.build.util {}).mkApplication {
        venv = virtualenv;
        package = pythonSet.audio-transcribe;
      };
      runtime-path = pkgs.lib.makeLibraryPath [pkgs.cudaPackages.cuda_cudart pkgs.cudaPackages.libcublas pkgs.cudaPackages.libcurand pkgs.cudaPackages.cudnn];
    in
      application.overrideAttrs (old: {
        nativeBuildInputs = old.nativeBuildInputs ++ [pkgs.installShellFiles] ++ pkgs.lib.optionals cudaSupport [pkgs.makeWrapper];
        postInstall = ''
          installShellCompletion --zsh ${self}/_audio-transcribe
        '';
        postFixup = pkgs.lib.optionalString cudaSupport ''
          # mkApplication symlinks the entry point into the immutable virtualenv.
          rm "$out/bin/audio-transcribe"
          makeWrapper ${virtualenv}/bin/audio-transcribe "$out/bin/audio-transcribe" \
            --prefix LD_LIBRARY_PATH : "${runtime-path}:/run/opengl-driver/lib:/usr/lib/wsl/lib"
        '';
        passthru =
          (old.passthru or {})
          // {
            inherit pythonSet virtualenv;
            sdk = pythonSet.audio-transcribe;
            cuda-library-path = runtime-path;
            tests.unit = pkgs.runCommand "audio-transcribe-unit" {} ''
              export PYTHONDONTWRITEBYTECODE=1
              ${virtualenv}/bin/python -m unittest discover -s ${self}/tests
              ${virtualenv}/bin/python -c 'import audio_transcribe, av, ctranslate2, faster_whisper, onnxruntime'
              touch "$out"
            '';
          };
        meta = (old.meta or {}) // {mainProgram = "audio-transcribe";};
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
    checks = each-system (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      unit = self.packages.${system}.default.tests.unit;
      uv-lock = pkgs.runCommand "audio-transcribe-uv-lock" {nativeBuildInputs = [pkgs.uv];} ''
        export UV_CACHE_DIR="$TMPDIR/uv-cache"
        export UV_PYTHON_DOWNLOADS=never
        cd ${self}
        uv lock --check --offline --python ${pkgs.python314}/bin/python
        touch "$out"
      '';
    });
    devShells = each-system (system: let
      pkgs = import nixpkgs {inherit system;};
    in {
      default = pkgs.mkShell {
        packages = [pkgs.uv pkgs.python314];
        env =
          {
            UV_PYTHON = pkgs.python314.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
          }
          // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.zlib];
          };
      };
    });
    formatter = each-system (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
