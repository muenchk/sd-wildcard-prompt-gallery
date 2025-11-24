// interaction functions for interacting


function wg_submit_s() {
    showSubmitButtons('txt2img', false);

    var id = randomId();
    localSet("txt2img_task_id", id);

    requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
        showSubmitButtons('txt2img', true);
        localRemove("txt2img_task_id");
        showRestoreProgressButton('txt2img', false);
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    return res;
}

function wg_submit() {
    showSubmitButtons('wg', false);

    var id = randomId();
    localSet("txt2img_task_id", id);

    requestProgress(id, gradioApp().getElementById('wg_gallery_container'), gradioApp().getElementById('wg_gallery'), function(){
        showSubmitButtons('wg', true);
        showRestoreProgressButton('wg', false);
    });

    requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
        showSubmitButtons('txt2img', true);
        localRemove("txt2img_task_id");
        showRestoreProgressButton('txt2img', false);
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    return res;
}

function restoreProgressWG() {
    showRestoreProgressButton("wg", false);
    showRestoreProgressButton("wg_prompt", false);
    var id = localGet("txt2img_task_id");

    if (id) {
        showSubmitInterruptingPlaceholder('wg');
        requestProgress(id, gradioApp().getElementById('wg_gallery_container'), gradioApp().getElementById('wg_gallery'), function() {
            showSubmitButtons('wg', true);
        }, null, 0);
    }
    if (id) {
        showSubmitInterruptingPlaceholder('wg_prompt');
        requestProgress(id, gradioApp().getElementById('wg_prompt_gallery_container'), gradioApp().getElementById('wg_prompt_gallery'), function() {
            showSubmitButtons('wg_prompt', true);
        }, null, 0);
    }
    if (id) {
        showSubmitInterruptingPlaceholder('txt2img');
        requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
            showSubmitButtons('txt2img', true);
        }, null, 0);
    }

    return id;
}


function wg_gal_submit() {
    showSubmitButtons('wg_prompt', false);

    var id = randomId();
    requestProgress(id, gradioApp().getElementById('wg_prompt_gallery_container'), gradioApp().getElementById('wg_prompt_gallery'), function(){
        showSubmitButtons('wg_prompt', true);
        showRestoreProgressButton('wg_prompt', false);
    });

    localSet("txt2img_task_id", id);
    requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
        showSubmitButtons('txt2img', true);
        localRemove("txt2img_task_id");
        showRestoreProgressButton('txt2img', false);
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    return res;
}

let wg_gallery;
let wg_prompt_gallery;

onAfterUiUpdate(function() {
    if (!wg_gallery) {
        wg_gallery = attachGalleryListeners("wg");
    }
    if (!wg_prompt_gallery) {
        wg_prompt_gallery = attachGalleryListeners("wg_prompt");
    }
});