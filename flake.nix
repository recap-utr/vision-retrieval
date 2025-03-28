{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
  };
  outputs =
    inputs@{
      flake-parts,
      systems,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import systems;
      perSystem =
        {
          pkgs,
          lib,
          ...
        }:
        {
          devShells.default = pkgs.mkShell {
            packages = with pkgs; [ uv ];
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.stdenv.cc.cc
              pkgs.zlib
            ];
            nativeBuildInputs = with pkgs; [
              zlib
            ];
            UV_PYTHON = lib.getExe pkgs.python312;
            shellHook = ''
              uv sync --all-extras --locked
            '';
          };
        };
    };
}
