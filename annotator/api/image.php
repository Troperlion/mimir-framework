<?php
require dirname(__DIR__) . '/config.php';
try {
    if (($_GET['dataset_root'] ?? null) !== root()) fail('Dataset actif différent.', 409);
    $p = image_path($_GET['image'] ?? '');
    $mime = (new finfo(FILEINFO_MIME_TYPE))->file($p);
    if (!in_array($mime, ['image/jpeg','image/png','image/webp','image/bmp','image/x-ms-bmp'], true)) fail('Format d’image invalide.', 415);
    session_write_close(); header('Content-Type: ' . $mime); header('X-Content-Type-Options: nosniff'); header('Cache-Control: no-store'); readfile($p);
} catch (Throwable $e) { if (http_response_code() < 400) http_response_code(400); echo htmlspecialchars($e->getMessage()); }
