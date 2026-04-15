// Updated render_article_block function
function render_article_block(data) {
    const container = document.createElement('div');

    // ... existing code to create chart ...

    // Create store selector
    const storeSelector = document.createElement('div');
    storeSelector.classList.add('store-selector');
    const stores = data.stores; // Assuming available stores are part of the data

    stores.forEach(store => {
        const storeButton = document.createElement('button');
        storeButton.textContent = store.name;
        storeButton.classList.add('store-button');
        storeButton.onclick = () => switchStore(store.id); // Function to switch store
        storeSelector.appendChild(storeButton);
    });

    container.appendChild(storeSelector);
    document.body.appendChild(container);
}

function switchStore(storeId) {
    // Logic to switch the displayed chart based on selected store
    console.log('Store switched to:', storeId);
    // ... Update chart accordingly ...
}