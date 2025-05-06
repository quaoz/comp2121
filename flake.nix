{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    uv2nix,
    pyproject-nix,
    pyproject-build-systems,
    ...
  }: let
    inherit (nixpkgs) lib;
    forAllSystems = lib.genAttrs lib.systems.flakeExposed;

    # Load a uv workspace from a workspace root.
    workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};

    # Create package overlay from workspace.
    overlay = workspace.mkPyprojectOverlay {
      sourcePreference = "wheel";
    };

    pyprojectOverrides = _final: _prev: {};

    # Construct package set
    pythonSet = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in
      # Use base package set from pyproject.nix builders
      (pkgs.callPackage pyproject-nix.build.packages {
        python = pkgs.python312;
      })
      .overrideScope
      (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
          pyprojectOverrides
        ]
      ));

    data = forAllSystems (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in
        pkgs.stdenv.mkDerivation rec {
          name = "nltk-data";

          punkt = pkgs.fetchzip {
            url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip";
            hash = "sha256-SKZu26K17qMUg7iCFZey0GTECUZ+sTTrF/pqeEgJCos=";
          };

          punkt-tab = pkgs.fetchzip {
            url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip";
            hash = "sha256-RwvF6O91YFg2DDMnykMOWQZCdmXAwfucHzkzwNHi3YY=";
          };

          stopwords = pkgs.fetchzip {
            url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip";
            hash = "sha256-jlxNhr0V2iLcY7RPbs3BZ2E9U2zABRVkLCeWRiSzvl4=";
          };

          wordnet = pkgs.fetchzip {
            url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip";
            hash = "sha256-L+lpHoDd9NwaDFBejPppF5hWg6e1+Sa9ixh3M4MzQs0=";
          };

          scifact = pkgs.fetchzip {
            url = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz";
            hash = "sha256-3Y1fGqm609po/4kmjwS1ly/kDJ2W5Z+x6HXS8EANCcE=";
            stripRoot = false;
          };

          dontUnpack = true;
          dontBuild = true;
          dontConfigure = true;

          installPhase = ''
            mkdir -p $out/nltk/{tokenizers,corpora}

            ln -s ${punkt} $out/nltk/tokenizers/punkt
            ln -s ${punkt-tab} $out/nltk/tokenizers/punkt_tab
            ln -s ${stopwords} $out/nltk/corpora/stopwords
            ln -s ${wordnet} $out/nltk/corpora/wordnet
            ln -s ${scifact}/data $out/scifact
          '';
        }
    );
  in {
    # Package a virtual environment as our main application.
    packages = forAllSystems (system: {
      default = pythonSet.${system}.mkVirtualEnv "comp2121-env" workspace.deps.default;
    });

    # Make hello runnable with `nix run`
    apps = forAllSystems (system: {
      default = {
        type = "app";
        program = "${self.packages.${system}.default}/bin/comp2121";
      };
    });

    devShells = forAllSystems (system: {
      default = let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
      in
        pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.ruff
          ];

          env = {
            UV_PYTHON_DOWNLOADS = "never";
            UV_PYTHON = python.interpreter;
            NLTK_DATA = "${data.${system}}/nltk";
            SCIFACT_DATA = "${data.${system}}/scifact";
          };
          shellHook = ''
            unset PYTHONPATH
          '';
        };
    });
  };
}
