#!/usr/bin/env bash
# Corre las pruebas de interfaz contra el Ollama real. Abren ventanas y usan la GPU.
# Una prueba que no imprime su resumen ("N/M OK") cuenta como falla: así un error al
# arrancar (archivo que falta, excepción) no pasa desapercibido.
cd "$(dirname "$0")"
PY="../../.venv/bin/python"
suites=(test_scripted test_ollama test_documents test_long)
[ "$#" -gt 0 ] && suites=("$@")
status=0
summary=()
for s in "${suites[@]}"; do
    echo "=== $s"
    out=$(timeout 3600 "$PY" "$s.py" 2>&1)
    code=$?
    echo "$out" | grep -E "^(OK|FALLA) |^      |/[0-9]+ OK$"
    if ! echo "$out" | grep -qE "^[0-9]+/[0-9]+ OK$"; then
        echo "$out" | tail -15
        summary+=("$s: NO TERMINÓ (código $code)")
        status=1
    elif [ "$code" -ne 0 ]; then
        summary+=("$s: $(echo "$out" | grep -E '^[0-9]+/[0-9]+ OK$') con fallas")
        status=1
    else
        summary+=("$s: $(echo "$out" | grep -E '^[0-9]+/[0-9]+ OK$')")
    fi
done
echo
printf '%s\n' "=== Resumen" "${summary[@]}"
exit $status
