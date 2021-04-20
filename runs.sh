#!/bin/sh
for dir in ../*; do
    { printf '%s\n' "$dir"
      ( cd "$dir" && find . )
    }
    # } >"$dir/original_filenames.txt"
done