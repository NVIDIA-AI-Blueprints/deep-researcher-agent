#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CI script to test all scripts in /scripts directory.
# Verifies syntax, help output, and basic functionality.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SCRIPTS_DIR="$REPO_ROOT/scripts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0
PASSED=0
SKIP_SETUP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-setup  Skip running setup.sh (use existing venv)"
            echo "  -h, --help    Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    PASSED=$((PASSED + 1))
}

log_fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    FAILED=$((FAILED + 1))
}

log_skip() {
    echo -e "${YELLOW}⏭️  SKIP${NC}: $1"
}

log_section() {
    echo ""
    echo "============================================"
    echo "  $1"
    echo "============================================"
    echo ""
}

# Test 1: Verify bash syntax for all scripts
test_bash_syntax() {
    log_section "Testing Bash Syntax"

    for script in "$SCRIPTS_DIR"/*.sh; do
        script_name=$(basename "$script")
        if bash -n "$script" 2>/dev/null; then
            log_pass "$script_name - valid bash syntax"
        else
            log_fail "$script_name - invalid bash syntax"
        fi
    done
}

# Test 2: Run setup.sh
test_setup() {
    log_section "Testing setup.sh"

    cd "$REPO_ROOT"

    if "$SCRIPTS_DIR/setup.sh"; then
        log_pass "setup.sh - completed successfully"
    else
        log_fail "setup.sh - failed to complete"
        echo "Cannot continue without setup.sh completing"
        exit 1
    fi
}

# Test 3: Test --help for scripts that support it
test_help_flags() {
    log_section "Testing --help Flags"

    # Scripts that support --help
    local help_scripts=("start_cli.sh" "start_server_in_debug_mode.sh")

    for script_name in "${help_scripts[@]}"; do
        script="$SCRIPTS_DIR/$script_name"
        if [ -f "$script" ]; then
            if "$script" --help >/dev/null 2>&1; then
                log_pass "$script_name --help"
            else
                log_fail "$script_name --help"
            fi
        else
            log_skip "$script_name - not found"
        fi
    done
}

# Test 4: Verify venv check in start scripts
test_venv_checks() {
    log_section "Testing Virtual Environment Checks"

    cd "$REPO_ROOT"

    # Temporarily rename .venv to test venv check
    if [ -d ".venv" ]; then
        mv .venv .venv_backup

        # Test start_cli.sh fails without venv
        if "$SCRIPTS_DIR/start_cli.sh" 2>&1 | grep -q "Virtual environment not found"; then
            log_pass "start_cli.sh - venv check works"
        else
            log_fail "start_cli.sh - venv check failed"
        fi

        # Test start_server_in_debug_mode.sh fails without venv
        if "$SCRIPTS_DIR/start_server_in_debug_mode.sh" 2>&1 | grep -q "Virtual environment not found"; then
            log_pass "start_server_in_debug_mode.sh - venv check works"
        else
            log_fail "start_server_in_debug_mode.sh - venv check failed"
        fi

        # Restore .venv
        mv .venv_backup .venv
    else
        log_skip "venv checks - .venv not found"
    fi
}

# Test 5: Test that pytest runs
test_pytest() {
    log_section "Testing Pytest Integration"

    cd "$REPO_ROOT"
    source "$REPO_ROOT/.venv/bin/activate"

    # Run a quick test to verify pytest works
    if pytest --collect-only -q 2>/dev/null | head -5; then
        log_pass "pytest collection works"
    else
        log_fail "pytest collection failed"
    fi
}

# Print summary
print_summary() {
    log_section "Test Summary"

    echo "Passed: $PASSED"
    echo "Failed: $FAILED"
    echo ""

    if [ $FAILED -gt 0 ]; then
        echo -e "${RED}Some tests failed!${NC}"
        exit 1
    else
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    fi
}

# Main
main() {
    echo ""
    echo "================================================"
    echo "  Deep Researcher Agent Blueprint - Script Tests"
    echo "================================================"
    echo ""
    echo "Repository: $REPO_ROOT"
    echo "Scripts:    $SCRIPTS_DIR"
    echo ""

    test_bash_syntax

    if [ "$SKIP_SETUP" = true ]; then
        log_skip "setup.sh - skipped (--skip-setup flag)"
    else
        test_setup
    fi

    test_help_flags
    test_venv_checks
    test_pytest

    print_summary
}

main "$@"
