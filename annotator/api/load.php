<?php
require dirname(__DIR__) . '/config.php';
api(function ($d) {
    $image = image_path($d['image'] ?? ''); $p = sidecar($image); $exists = is_file($p);
    $a = $exists ? read_json($p) : blank_annotation($image);
    // Never silently turn an unrelated JSON into an annotation.
    if ($exists) validate_annotation($a, $image);
    return ['annotation'=>$a, 'exists'=>$exists, 'revision'=>$exists ? hash_file('sha256',$p) : null];
});
