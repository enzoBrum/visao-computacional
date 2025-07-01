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

            cd $HOME/ufsc/ine5448
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
            (python312.override {
              x11Support = true;
            })
            python311
            python310
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
            libGLU libGL
            xorg.libXi xorg.libXmu freeglut
            xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
            ncurses5
            stdenv.cc
            binutils
            opencv
            #linuxPackages.nvidia_x11_latest
            #cudaPackages.cudnn
            #cudatoolkit
            qt5.full
            qt5.qtbase
            xorg.libxcb
            freetype
            xorg.libSM
            xorg.libICE

              #(opencv.override {
            #  enableGtk3 = true;
            #  enableCuda = true;
            #  enableUnfree = true;
            #  enableFfmpeg = true;
            #})
          ]);
          profile = ''QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms" SHELL=/usr/bin/my-fish NIXOS_OZONE_WL="" EXTRA_LDFLAGS="-L/lib" EXTRA_CCFLAGS="-I/usr/include" my-fish'';
        }).env;
      });
}

