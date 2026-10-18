<?php
require dirname(__DIR__) . '/config.php';
api(function ($d) {
    $relative = $d['image'] ?? ''; $image = image_path($relative); $p = sidecar($image);
    $a = $d['annotation'] ?? []; if (!is_array($a)) fail('Annotation invalide.'); validate_annotation($a, $image);
    return with_lock(function () use ($d, $p, $a, $relative) {
        $revision = is_file($p) ? hash_file('sha256', $p) : null;
        if (!array_key_exists('revision', $d) || $d['revision'] !== $revision) fail('Ce JSON a changé depuis son chargement. Rechargez l’image avant de sauvegarder.', 409);
        atomic_write($p, $a); $warning = null;
        try { $state = update_state($relative, true); }
        catch (Throwable $e) { $state = null; $warning = 'JSON sauvegardé, mais historique non enregistré : ' . $e->getMessage(); http_response_code(200); }
        return ['saved'=>true,'revision'=>hash_file('sha256',$p),'state'=>$state,'warning'=>$warning];
    });
});
