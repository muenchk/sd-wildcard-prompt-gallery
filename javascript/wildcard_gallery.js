// interaction functions for interacting

function wg_submit() {
    showSubmitButtons('wg', false);

    var id = randomId();
    requestProgress(id, gradioApp().getElementById('wg_gallery_container'), gradioApp().getElementById('wg_gallery'), function(){
        showSubmitButtons('wg', true);
        showRestoreProgressButton('txt2img', false);
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

function restoreProgressWG() {
    showRestoreProgressButton("wg", false);
    var id = localGet("txt2img_task_id");

    if (id) {
        showSubmitInterruptingPlaceholder('txt2img');
        requestProgress(id, gradioApp().getElementById('wg_gallery_container'), gradioApp().getElementById('wg_gallery'), function() {
            showSubmitButtons('wg', true);
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

let wg_gallery;

onAfterUiUpdate(function() {
    if (!wg_gallery) {
        wg_gallery = attachGalleryListeners("wg");
    }
});