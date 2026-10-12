{pkgs}: final: prev: {
  ctranslate2 = prev.ctranslate2.overrideAttrs (old: {
    buildInputs = (old.buildInputs or []) ++ [pkgs.stdenv.cc.cc.lib];
  });
}
