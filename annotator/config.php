<?php
declare(strict_types=1);
// Local-only tool: launch with php -S 127.0.0.1:8080 -t annotator.
if (!in_array($_SERVER['REMOTE_ADDR'] ?? '', ['127.0.0.1', '::1'], true)) {
    http_response_code(403); exit('Accès local uniquement.');
}
// Reject DNS rebinding: a remote hostname must not access the local filesystem API.
if (!preg_match('/^(localhost|127\.0\.0\.1|\[::1\])(:[0-9]+)?$/i', $_SERVER['HTTP_HOST'] ?? '')) {
    http_response_code(403); exit('Hôte local requis.');
}
session_name('lumen_annotator');
session_start(['cookie_httponly' => true, 'cookie_samesite' => 'Strict']);
$_SESSION['csrf'] ??= bin2hex(random_bytes(32));
function fail(string $message, int $code = 400): void { http_response_code($code); throw new RuntimeException($message); }
function respond(array $value): void { header('Content-Type: application/json; charset=utf-8'); echo json_encode($value, JSON_UNESCAPED_UNICODE | JSON_THROW_ON_ERROR); }
function request(): array {
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') fail('POST requis.', 405);
    if (!hash_equals($_SESSION['csrf'], $_SERVER['HTTP_X_CSRF_TOKEN'] ?? '')) fail('Jeton de session invalide. Rechargez la page.', 403);
    $body = file_get_contents('php://input');
    if (strlen($body) > 4 * 1024 * 1024) fail('Annotation trop volumineuse.', 413);
    $data = json_decode($body, true, 128, JSON_THROW_ON_ERROR);
    if (!is_array($data)) fail('Objet JSON requis.');
    if (basename($_SERVER['SCRIPT_NAME']) !== 'scan.php' && ($data['dataset_root'] ?? null) !== root()) fail('Le dataset actif a changé dans un autre onglet. Rouvrez votre dossier.', 409);
    return $data;
}
function root(): string { $r = $_SESSION['dataset_root'] ?? null; if (!$r || !is_dir($r)) fail('Ouvrez un dossier de dataset.', 409); return $r; }
function inside(string $path, string $r): bool { return str_starts_with($path, $r . DIRECTORY_SEPARATOR); }
function image_path(string $relative): string {
    $r = root();
    if ($relative === '' || str_contains($relative, '\\') || str_starts_with($relative, '/') || in_array('..', explode('/', $relative), true)) fail('Chemin interdit.', 403);
    $lexical = $r . '/' . $relative; $real = realpath($lexical);
    if (!$real || !inside($real, $r) || !is_file($real) || !in_array(strtolower(pathinfo($lexical, PATHINFO_EXTENSION)), ['jpg','jpeg','png','webp','bmp'], true)) fail('Image inaccessible.', 404);
    $parent = realpath(dirname($lexical)); if (!$parent || !inside($parent . '/', $r)) fail('Dossier interdit.', 403);
    return $lexical;
}
function sidecar(string $image): string {
    $path = substr($image, 0, strrpos($image, '.')) . '.json';
    if (is_link($path) || (file_exists($path) && !is_file($path))) fail('Le JSON cible doit être un fichier ordinaire.', 409);
    // Cache sibling stems per request so recursive scans stay linear.
    static $stems = [];
    $directory = dirname($image);
    if (!isset($stems[$directory])) {
        $stems[$directory] = [];
        foreach (new DirectoryIterator($directory) as $entry) {
            if ($entry->isFile() && in_array(strtolower($entry->getExtension()), ['jpg','jpeg','png','webp','bmp'], true)) {
                $stem = pathinfo($entry->getFilename(), PATHINFO_FILENAME);
                $stems[$directory][$stem] = ($stems[$directory][$stem] ?? 0) + 1;
            }
        }
    }
    if (($stems[$directory][pathinfo($image, PATHINFO_FILENAME)] ?? 0) > 1) fail('Plusieurs images partagent ce nom sans extension. Renommez-les avant annotation.', 409);
    return $path;
}
function atomic_write(string $path, array $data): void {
    if (is_link($path)) fail('Lien symbolique de sortie interdit.', 409);
    $json = json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR) . "\n";
    $tmp = tempnam(dirname($path), '.annotation-tmp-');
    if (!$tmp) fail('Impossible de créer le fichier temporaire.', 500);
    try { if (file_put_contents($tmp, $json, LOCK_EX) !== strlen($json) || !rename($tmp, $path)) fail('Écriture impossible. Vérifiez les permissions.', 500); }
    finally { if (file_exists($tmp)) unlink($tmp); }
}
function read_json(string $path): array {
    $data = json_decode(file_get_contents($path), true, 128, JSON_THROW_ON_ERROR);
    if (!is_array($data)) fail('JSON existant invalide : ' . basename($path), 409);
    return $data;
}
function state_path(): string { $p = root() . '/.annotation_state.json'; if (is_link($p)) fail('Historique symbolique interdit.', 409); return $p; }
function state_read(): array { $p = state_path(); return is_file($p) ? read_json($p) : []; }
function with_lock(callable $fn) {
    $p = root() . '/.annotation_state.lock'; if (is_link($p)) fail('Verrou symbolique interdit.', 409);
    $h = fopen($p, 'c'); if (!$h || !flock($h, LOCK_EX)) fail('Verrou impossible.', 500);
    try { return $fn(); } finally { flock($h, LOCK_UN); fclose($h); }
}
function update_state(string $relative, bool $saved): array {
    image_path($relative); $s = state_read(); $images = $_SESSION['images'] ?? [];
    $index = array_search($relative, $images, true); if ($index === false) fail('Relancez le scan.', 409);
    $s['dataset_root'] = root(); $s['last_image'] = $relative; $s['current_index'] = $index;
    $s['annotated'] = array_values(array_intersect($images, $s['annotated'] ?? []));
    if ($saved) { $s['annotated'] = array_values(array_unique(array_merge($s['annotated'], [$relative]))); $s['last_annotation'] = $relative; }
    $s['updated_at'] = date(DATE_ATOM); atomic_write(state_path(), $s); return $s;
}
function blank_annotation(string $image): array {
    return ['image'=>basename($image),'captions'=>['primary'=>'','alternatives'=>[]], 'entities'=>[], 'attributes'=>[], 'relations'=>[],
        'lighting'=>['description'=>'','source'=>'','target'=>''], 'composition'=>['description'=>''], 'quality'=>['caption_verified'=>false,'relations_verified'=>false]];
}
function validate_annotation(array $a, string $image): void {
    if (($a['image'] ?? null) !== basename($image)) fail('Nom de l’image incompatible.');
    if (!is_string($a['captions']['primary'] ?? null) || trim($a['captions']['primary']) === '') fail('Une description naturelle principale est requise.');
    foreach (['entities','attributes','relations'] as $key) if (!isset($a[$key]) || !is_array($a[$key]) || !array_is_list($a[$key])) fail('Liste invalide : ' . $key);
    if (!is_array($a['captions']['alternatives'] ?? null) || !array_is_list($a['captions']['alternatives'])) fail('Alternatives invalides.');
    foreach ($a['captions']['alternatives'] as $v) if (!is_string($v)) fail('Alternative invalide.');
    $ids = [];
    foreach ($a['entities'] as $e) {
        if (!is_array($e) || !is_string($e['id'] ?? null) || trim($e['id']) === '' || isset($ids[$e['id']]) || !is_string($e['phrase'] ?? null) || !is_string($e['role'] ?? null)) fail('Entité invalide ou identifiant dupliqué.');
        $ids[$e['id']] = true;
        if (array_key_exists('skeleton', $e)) {
            $s = $e['skeleton'];
            if (!is_array($s) || !is_string($s['name'] ?? null) || trim($s['name']) === '' || !is_array($s['points'] ?? null) || !array_is_list($s['points'])) fail('Squelette invalide.');
            $names = [];
            foreach ($s['points'] as $point) {
                if (!is_array($point) || !is_string($point['name'] ?? null) || trim($point['name']) === '' || isset($names[$point['name']])) fail('Nom de point vide ou dupliqué.');
                $names[$point['name']] = true;
                if (array_key_exists('role', $point) && !in_array($point['role'], ['point','tête','support','saisie','articulation majeure','détail facial'], true)) fail('Nature de point invalide.');
                if (!array_key_exists('position', $point)) fail('Position de point manquante.');
                if ($point['position'] !== null) {
                    $pos = $point['position'];
                    if (!is_array($pos) || !array_is_list($pos) || count($pos) !== 2) fail('Position de point invalide.');
                    foreach ($pos as $n) if ((!is_int($n) && !is_float($n)) || !is_finite((float)$n) || $n < 0 || $n > 1) fail('Point hors de [0,1].');
                }
                if (isset($point['parent']) && !is_string($point['parent'])) fail('Liaison de squelette invalide.');
            }
            if (array_key_exists('edges', $s)) {
                if (!is_array($s['edges']) || !array_is_list($s['edges'])) fail('Liste de liaisons invalide.');
                foreach ($s['edges'] as $edge) {
                    if (!is_array($edge) || !is_string($edge['from'] ?? null) || !is_string($edge['to'] ?? null) || !isset($names[$edge['from']], $names[$edge['to']]) || $edge['from'] === $edge['to'] || (isset($edge['label']) && !is_string($edge['label']))) fail('Liaison libre invalide.');
                }
            }
            foreach ($s['points'] as $point) if (isset($point['parent']) && (!isset($names[$point['parent']]) || $point['parent'] === $point['name'])) fail('Liaison de squelette inconnue ou vers soi-même.');
        }
        if (isset($e['bbox'])) {
            $b = $e['bbox']; if (!is_array($b) || !array_is_list($b) || count($b) !== 4) fail('BBox invalide.');
            foreach ($b as $n) if ((!is_int($n) && !is_float($n)) || !is_finite((float)$n) || $n < 0 || $n > 1) fail('BBox hors de [0,1].');
            if ($b[0] >= $b[2] || $b[1] >= $b[3]) fail('BBox vide ou inversée.');
        }
    }
    foreach ($a['entities'] as $e) foreach ($e['skeleton']['points'] ?? [] as $point) {
        if (!array_key_exists('entities', $point)) continue;
        if (!is_array($point['entities']) || !array_is_list($point['entities'])) fail('Liste des entités liées invalide.');
        $linked = [];
        foreach ($point['entities'] as $id) {
            if (!is_string($id) || !isset($ids[$id]) || isset($linked[$id])) fail('Entité liée inconnue ou dupliquée.');
            $linked[$id] = true;
        }
    }
    foreach ($a['attributes'] as $v) if (!isset($ids[$v['entity'] ?? '']) || !is_string($v['property'] ?? null) || !is_string($v['value'] ?? null)) fail('Attribut invalide.');
    foreach ($a['relations'] as $v) if (!isset($ids[$v['subject'] ?? ''], $ids[$v['object'] ?? '']) || !is_string($v['predicate'] ?? null) || trim($v['predicate']) === '') fail('Relation invalide.');
    foreach (['description','source','target'] as $key) if (!is_string($a['lighting'][$key] ?? null)) fail('Éclairage invalide.');
    if (!is_string($a['composition']['description'] ?? null)) fail('Composition invalide.');
    foreach (['caption_verified','relations_verified'] as $key) if (!is_bool($a['quality'][$key] ?? null)) fail('Qualité invalide.');
}
function api(callable $fn): void {
    try { respond($fn(request())); }
    catch (Throwable $e) { if (http_response_code() < 400) http_response_code(400); respond(['error'=>$e->getMessage()]); }
}
