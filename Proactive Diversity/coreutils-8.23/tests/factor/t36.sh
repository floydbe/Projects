#!/bin/sh

# === THIS IS A TEST SCRIPT DERIVED FROM THE TEMPLATE ./tests/factor/run.sh ===

# Test the factor rewrite.
# The test is to run this command
# seq $START $END | factor | shasum -c --status <(echo $CKSUM  -)
# I.e., to ensure that the factorizations of integers $1..$2
# match what we expect.
#
# See: tests/factor/create-test.sh

# Copyright (C) 2012-2014 Free Software Foundation, Inc.

. "${srcdir=.}/tests/init.sh"; path_prepend_ ./src

# Don't run these tests by default.
very_expensive_

print_ver_ factor seq sha1sum

# Template variables.
START=79228162514264337593543960336
  END=79228162514264337593543962335
CKSUM=569e4363e8d9e8830a187d9ab27365eef08abde1

echo "$CKSUM  -" > exp

f=1
seq $START $END | factor | sha1sum -c --status exp && f=0

Exit $f
