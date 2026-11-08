"""
Ponte de integração entre o cent_runtime e o simulador de ciclo exato Ramulator 2.0.
Invoca o executável ramulator2 (aim_simulator) consumindo arquivos .trace gerados
pela CENT ISA e extrai as métricas de latência e contagem de ciclos dos bancos AiM.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from ._bindings import _find_repo_root


def resolve_ramulator_bin(ramulator_bin: Optional[str] = None) -> Path:
    """
    Resolve o caminho do executável do simulador Ramulator 2.0 respeitando
    a seguinte ordem de prioridade portátil:
      1. Argumento explícito 'ramulator_bin';
      2. Variável de ambiente 'CENT_RAMULATOR_BIN';
      3. Caminho relativo irmão: <repo_root>/../CENT/aim_simulator/build/ramulator2;
      4. Busca no PATH do sistema operacional (shutil.which).
    """
    if ramulator_bin:
        path = Path(ramulator_bin).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise FileNotFoundError(f"Executável ramulator2 especificado não é válido: {ramulator_bin}")

    env_bin = os.environ.get("CENT_RAMULATOR_BIN")
    if env_bin:
        path = Path(env_bin).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise FileNotFoundError(f"Variável CENT_RAMULATOR_BIN aponta para executável inválido: {env_bin}")

    repo_root = _find_repo_root()
    sibling_path = (repo_root.parent / "CENT" / "aim_simulator" / "build" / "ramulator2").resolve()
    if sibling_path.is_file() and os.access(sibling_path, os.X_OK):
        return sibling_path

    which_path = shutil.which("ramulator2")
    if which_path:
        return Path(which_path).resolve()

    raise FileNotFoundError(
        "Binário do Ramulator 2.0 (ramulator2) não foi encontrado. "
        "Especifique via argumento 'ramulator_bin', configure a variável "
        "de ambiente 'CENT_RAMULATOR_BIN', ou compile em ../CENT/aim_simulator/build/ramulator2."
    )


def resolve_config_path(config_path: Optional[str] = None) -> Path:
    """
    Resolve o caminho do arquivo de configuração YAML para o Ramulator 2.0:
      1. Argumento explícito 'config_path';
      2. Variável de ambiente 'CENT_RAMULATOR_CONFIG';
      3. Padrão no repositório: <repo_root>/configs/gddr6_aim.yaml.
    """
    if config_path:
        path = Path(config_path).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Arquivo de configuração especificado não encontrado: {config_path}")

    env_cfg = os.environ.get("CENT_RAMULATOR_CONFIG")
    if env_cfg:
        path = Path(env_cfg).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Variável CENT_RAMULATOR_CONFIG aponta para arquivo inválido: {env_cfg}")

    repo_root = _find_repo_root()
    default_cfg = repo_root / "configs" / "gddr6_aim.yaml"
    if default_cfg.is_file():
        return default_cfg

    raise FileNotFoundError(f"Arquivo padrão de configuração não encontrado em: {default_cfg}")


def parse_simulator_output(stdout_text: str) -> Dict[str, Any]:
    """
    Faz o parse estruturado da saída textual padrão do Ramulator 2.0.
    Ignora linhas de comentários (#) e converte pares chave: valor numéricos.
    """
    parsed: Dict[str, Any] = {}
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip()
            val_str = parts[1].strip()

            if not val_str:
                continue

            try:
                if "." in val_str or "e" in val_str.lower():
                    parsed[key] = float(val_str)
                else:
                    parsed[key] = int(val_str)
            except ValueError:
                parsed[key] = val_str

    wr_gb = parsed.get("CH31_AiM_ISR_WR_GB_cycles", parsed.get("CH0_AiM_ISR_WR_GB_cycles", 0))
    mac_abk = parsed.get("CH31_AiM_ISR_MAC_ABK_cycles", parsed.get("CH0_AiM_ISR_MAC_ABK_cycles", 0))
    mem_cycles = parsed.get("memory_system_cycles", 0)

    return {
        "status": "success",
        "wr_gb_cycles": wr_gb,
        "mac_abk_cycles": mac_abk,
        "memory_system_cycles": mem_cycles,
        "parsed_stats": parsed,
    }


def simulate_on_ramulator(
    trace_path: str,
    config_path: Optional[str] = None,
    ramulator_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Dispara uma simulação de ciclo exato no Ramulator 2.0 usando a configuração
    GDDR6-AiM e o arquivo .trace fornecido.
    """
    trace_file = Path(trace_path).expanduser().resolve()
    if not trace_file.is_file():
        raise FileNotFoundError(f"Arquivo de traço CENT ISA (.trace) não encontrado: {trace_path}")

    bin_path = resolve_ramulator_bin(ramulator_bin)
    cfg_path = resolve_config_path(config_path)

    cmd = [str(bin_path), "-f", str(cfg_path), "-t", str(trace_file)]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as err:
        stderr_msg = err.stderr.strip() if err.stderr else ""
        raise RuntimeError(
            f"Erro na execução do Ramulator 2.0 (código {err.returncode}): {stderr_msg}"
        ) from err

    return parse_simulator_output(proc.stdout)
