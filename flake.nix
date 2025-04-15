{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {inherit system;};
        fish-iek = pkgs.writeScriptBin "fish-iek" /* bash */ ''
            fish -C "
            if not test -d .venv
              python -m venv .venv
              source .venv/bin/activate.fish

              pip install uv
              deactivate
            end

            source .venv/bin/activate.fish
            cd -
            "

        '';
      in {
        devShells.default = (pkgs.buildFHSEnv {
          name = "labsec";
          targetPkgs = pkgs: (with pkgs; [
            gcc
            python312
            stdenv.cc.cc.lib
            openssl
            zlib
            fish
            fish-iek
          ]);
          profile = "SHELL=/usr/bin/fish-iek fish-iek";
        }).env;
      });
}

