{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  buildInputs = [
    (python3.withPackages (pp: with pp; [
      pymupdf
      pillow
      nur.repos.milahu.python3.pkgs.doxapy
    ]))
  ];
}
