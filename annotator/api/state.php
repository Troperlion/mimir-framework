<?php
require dirname(__DIR__) . '/config.php';
api(function ($d) { return with_lock(function () use ($d) { return ['state'=>isset($d['image']) ? update_state($d['image'], false) : state_read()]; }); });
