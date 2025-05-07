{
  inputs = {
    # TODO: revert once https://github.com/NixOS/nixpkgs/pull/404881 in merged
    nixpkgs.url = "github:hoh/nixpkgs/hoh-fix-spacy";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    inherit (nixpkgs) lib;
    forAllSystems = function:
      lib.genAttrs lib.systems.flakeExposed (
        system: function nixpkgs.legacyPackages.${system}
      );

    spacy-models = pkgs: let
      model = pname: version: hash:
        pkgs.python312Packages.buildPythonPackage {
          inherit pname version;

          src = pkgs.fetchzip {
            url = "https://github.com/explosion/spacy-models/releases/download/${pname}-${version}/${pname}-${version}.tar.gz";
            inherit hash;
          };
        };

      version = "3.8.0";
    in rec {
      en_core_web_sm = model "en_core_web_sm" version "sha256-zLQcu0sK6wec3COjxVa+oqP91EGYf1OCz1i4KHZh44I=";
      en_core_web_md = model "en_core_web_md" version "sha256-0+W2x+xUYrHs4e+EibhoRcxXMfC8SnUXVK1Lh/RiIaU=";
      en_core_web_lg = model "en_core_web_lg" version "sha256-/Wz+cuS62Q9Z6QYvsz7CCK9vkcaz8DFKRUFy4HZXfI8=";

      all = [en_core_web_sm en_core_web_md en_core_web_lg];
    };

    data = pkgs:
      pkgs.stdenv.mkDerivation rec {
        name = "data";

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
      };

    pythonSet = pkgs:
      pkgs.python312.withPackages (ps:
        with ps;
          [
            hf-xet
            ipython
            matplotlib
            nltk
            numpy
            optuna
            pandas
            scikit-learn
            seaborn
            sentence-transformers
            spacy
            textstat
            torch
            tqdm
            transformers
            jupyter
          ]
          ++ (spacy-models pkgs).all);
  in {
    packages = forAllSystems (pkgs: {
      default = self.packages.${pkgs.system}.lab;

      lab = pkgs.writeShellScriptBin "launch-lab" ''
        exec ${pythonSet pkgs}/bin/python -m jupyter lab ${self}/src/pipeline.ipynb
      '';

      notebook = pkgs.writeShellScriptBin "launch-notebook" ''
        exec ${pythonSet pkgs}/bin/python -m jupyter lab ${self}/src/pipeline.ipynb
      '';
    });

    apps = forAllSystems (pkgs: {
      default = {
        type = "app";
        program = "${self.packages.${pkgs.system}.default}/bin/launch-lab";
      };
    });

    devShells = forAllSystems (pkgs: {
      default = pkgs.mkShell {
        packages = [
          (pythonSet pkgs)
          pkgs.ruff
        ];

        env = {
          NLTK_DATA = "${data pkgs}/nltk";
          SCIFACT_DATA = "${data pkgs}/scifact";
        };
      };
    });
  };
}
