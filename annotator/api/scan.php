<?php
require dirname(__DIR__) . '/config.php';
api(function ($d) {
    $previousRoot = $_SESSION['dataset_root'] ?? null; $previousImages = $_SESSION['images'] ?? null;
    try {
    $r = realpath($d['root'] ?? ''); if (!$r || !is_dir($r) || !is_readable($r) || $r === '/') fail('Choisissez un dossier lisible, autre que la racine système.');
    $_SESSION['dataset_root'] = $r; $images = []; $annotated = []; $warnings = [];
    $it = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($r, FilesystemIterator::SKIP_DOTS), RecursiveIteratorIterator::LEAVES_ONLY, RecursiveIteratorIterator::CATCH_GET_CHILD);
    foreach ($it as $f) {
        if (!$f->isFile() || !in_array(strtolower($f->getExtension()), ['jpg','jpeg','png','webp','bmp'], true)) continue;
        $relative = substr($f->getPathname(), strlen($r) + 1);
        try { $image = image_path($relative); } catch (Throwable $e) { $warnings[] = $relative . ' : ' . $e->getMessage(); http_response_code(200); continue; }
        $images[] = $relative;
        try { $p = sidecar($image); if (is_file($p)) { $a = read_json($p); validate_annotation($a, $relative); $annotated[] = $relative; } }
        catch (Throwable $e) { $warnings[] = $relative . ' : ' . $e->getMessage(); http_response_code(200); }
    }
    sort($images, SORT_NATURAL); $_SESSION['images'] = $images;
    $s = with_lock(function () use ($annotated) { $s = state_read(); $s['annotated'] = $annotated; $s['dataset_root'] = root(); $s['updated_at'] = date(DATE_ATOM); atomic_write(state_path(), $s); return $s; });
    $last = array_search($s['last_image'] ?? '', $images, true); $resume = $last === false ? 0 : $last;
    if (isset($images[$resume]) && in_array($images[$resume], $annotated, true)) {
        for ($offset = 1; $offset <= count($images); $offset++) { $i = ($resume + $offset) % count($images); if (!in_array($images[$i], $annotated, true)) { $resume = $i; break; } }
    }
    return ['root'=>$r,'images'=>$images,'annotated'=>$annotated,'state'=>$s,'resume_index'=>$resume,'warnings'=>$warnings];
    } catch (Throwable $e) {
        if ($previousRoot === null) unset($_SESSION['dataset_root']); else $_SESSION['dataset_root'] = $previousRoot;
        if ($previousImages === null) unset($_SESSION['images']); else $_SESSION['images'] = $previousImages;
        throw $e;
    }
});
