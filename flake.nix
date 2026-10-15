{
  description = "Whisper model management, transcription CLI and Python SDK";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  outputs = {
    self,
    nixpkgs,
  }: let
    systems = ["x86_64-linux" "aarch64-linux" "aarch64-darwin"];
    each-system = nixpkgs.lib.genAttrs systems;
  in {
    lib.mkPackage = {
      pkgs,
      cudaSupport ? false,
      python ? pkgs.python314,
    }: let
      ctranslate2-cpp = pkgs.ctranslate2.override {
        withCUDA = cudaSupport;
        withCuDNN = cudaSupport;
      };
      ctranslate2 = python.pkgs.ctranslate2.override {inherit ctranslate2-cpp;};
      faster-whisper = python.pkgs.faster-whisper.override {inherit ctranslate2;};
      sdk = python.pkgs.buildPythonPackage {
        pname = "audio-transcribe";
        version = "0.2.0";
        src = self;
        pyproject = true;
        build-system = [python.pkgs.uv-build];
        dependencies = [faster-whisper python.pkgs.huggingface-hub python.pkgs.filelock];
        nativeBuildInputs = [pkgs.installShellFiles];
        postInstall = "installShellCompletion --zsh _audio-transcribe";
        pythonImportsCheck = ["audio_transcribe"];
        passthru.tests.unit = pkgs.runCommand "audio-transcribe-unit" {} ''
          export PYTHONPATH=${self}/src
          export PYTHONDONTWRITEBYTECODE=1
          ${python.withPackages (ps: [ps.filelock])}/bin/python -m unittest discover -s ${self}/tests
          touch "$out"
        '';
      };
      runtime-path = pkgs.lib.makeLibraryPath [pkgs.cudaPackages.cuda_cudart pkgs.cudaPackages.libcublas pkgs.cudaPackages.libcurand pkgs.cudaPackages.cudnn];
    in
      sdk.overrideAttrs (old: {
        nativeBuildInputs = old.nativeBuildInputs ++ pkgs.lib.optionals cudaSupport [pkgs.makeWrapper];
        postFixup =
          (old.postFixup or "")
          + pkgs.lib.optionalString cudaSupport ''
            wrapProgram "$out/bin/audio-transcribe" --prefix LD_LIBRARY_PATH : "${runtime-path}:/run/opengl-driver/lib:/usr/lib/wsl/lib"
          '';
        passthru = old.passthru // {cuda-library-path = runtime-path;};
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
    checks = each-system (system: {unit = self.packages.${system}.default.tests.unit;});
    devShells = each-system (system: let
      pkgs = import nixpkgs {inherit system;};
    in {
      default = pkgs.mkShell {
        packages = [pkgs.uv pkgs.python314];
        env = pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.zlib];
        };
      };
    });
    formatter = each-system (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
