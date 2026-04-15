def render_article_block(article):
    # Other existing code here...

    # Display the Plotly chart
    st.plotly_chart(fig)

    # Adding store selection buttons below the chart
    available_stores = get_available_stores(article)  # Assuming you have a function to get store data
    selected_stores = st.multiselect('Select Stores:', available_stores, default=available_stores)

    # Create buttons for each store
    store_buttons = []
    for store in available_stores:
        is_selected = store in selected_stores
        button_color = 'primary' if is_selected else 'light'
        store_buttons.append(st.button(store, key=store, style=f'background-color: {button_color};'))

    # Select All and Clear buttons
    if st.button('Select All'):
        selected_stores = available_stores
    if st.button('Clear'):
        selected_stores = []

    # Save the selection
    st.session_state.selected_stores = selected_stores
    # Other existing code continues...