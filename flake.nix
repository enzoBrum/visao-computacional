{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {inherit system; config.allowUnfree = true; config.cudaSupport = true;};
        my-fish = pkgs.writeScriptBin "my-fish" /* bash */ ''
            fish -C "

            cd $HOME/ufsc/INE5448
            if not test -d .venv
              python -m venv .venv
              source .venv/bin/activate.fish

              pip install uv
              if test -f requirements.txt
                uv pip install -r requirements.txt
              end
              deactivate
            end

            source .venv/bin/activate.fish
            cd -
            "
        '';
      in {
        devShells.default = (pkgs.buildFHSEnv {
          name = "INE410121";
          targetPkgs = pkgs: (with pkgs; [
            gcc
            python313Full
            python313Packages.pip
            python313Packages.virtualenv
            uv
            openssl
            zlib
            fish
            my-fish
            cmake
            ninja
            libGL
            libglvnd
            mesa
            glib
            git
            gitRepo
            gnupg
            autoconf
            curl
            procps
            gnumake
            util-linux
            m4
            gperf
            unzip
            cudatoolkit
            linuxPackages.nvidia_x11
            libGLU libGL
            xorg.libXi xorg.libXmu freeglut
            xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
            ncurses5
            stdenv.cc
            binutils
            opencv
            cacert
            #(opencv.override {
            #  enableGtk3 = true;
            #  enableCuda = true;
            #  enableUnfree = true;
            #  enableFfmpeg = true;
            #})
          ]);
          profile = ''SHELL=/usr/bin/my-fish CUDA_PATH=${pkgs.cudatoolkit} EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib" EXTRA_CCFLAGS="-I/usr/include" my-fish'';
        }).env;
      });
}
