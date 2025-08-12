const chatInner = document.getElementById('chatInner');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');

let currentChatId = null;

function addMessage(role, text, id = null) {
  const row = document.createElement('div');
  row.className = 'message-row ' + role;
  if (id) row.id = id;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (text.includes('<') && text.includes('>')) {
    bubble.innerHTML = text;
  } else {
    bubble.textContent = text;
  }
  row.appendChild(bubble);
  chatInner.appendChild(row);
  chatInner.scrollTop = chatInner.scrollHeight;
}

function displayPropertyListings(listings) {
  if (!listings || listings.length === 0) return;
  
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  
  let listingsHtml = '<div style="margin-top: 10px;"><strong>Property Listings:</strong><br><br>';
  
  listings.forEach((listing, index) => {
    listingsHtml += `<div style="margin-bottom: 40px; padding: 15px; border-left: 3px solid var(--text); background: var(--bg-alt); border-radius: 8px;">`;
    
    // Title
    const cleanTitle = (listing.title || 'Property').replace(' | Graana.com', '').split('|')[0].trim();
    listingsHtml += `<strong style="font-size: 1.1em; color: var(--text);">${index + 1}. ${cleanTitle}</strong><br><br>`;
    
    // Price
    if (listing.price && listing.price !== 'Price not available') {
      listingsHtml += `<div style="margin: 5px 0;"><strong>💰 Price:</strong> ${listing.price}</div>`;
    }
    
    // Bedrooms
    if (listing.bedrooms && listing.bedrooms !== "not provided") {
      listingsHtml += `<div style="margin: 5px 0;"><strong>🛏️ Bedrooms:</strong> ${listing.bedrooms}</div>`;
    }
    
    // Kitchens
    if (listing.kitchens && listing.kitchens !== "not provided") {
      listingsHtml += `<div style="margin: 5px 0;"><strong>🍳 Kitchens:</strong> ${listing.kitchens}</div>`;
    }
    
    // Area
    if (listing.area) {
      listingsHtml += `<div style="margin: 5px 0;"><strong>📐 Area:</strong> ${listing.area}</div>`;
    }
    
    // Location
    if (listing.location) {
      listingsHtml += `<div style="margin: 5px 0;"><strong>📍 Location:</strong> ${listing.location}</div>`;
    }
    
    // Description (if available and not empty)
    if (listing.description && listing.description.trim() && listing.description !== "not provided") {
      const shortDesc = listing.description.length > 150 ? listing.description.substring(0, 150) + '...' : listing.description;
      listingsHtml += `<div style="margin: 8px 0; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 6px;">`;
      listingsHtml += `<strong>📝 Description:</strong><br>${shortDesc}`;
      listingsHtml += `</div>`;
    }
    
    // Images button
    if (listing.images && listing.images.length > 0) {
      listingsHtml += `<div style="margin: 30px 0;">`;
      listingsHtml += `<button onclick="showImagePopup('${listing.property_id}', ${JSON.stringify(listing.images).replace(/"/g, '&quot;')})" 
           style="background: var(--link); color: white; border: none; padding: 8px 12px; border-radius: 5px; cursor: pointer; font-weight: 500; font-size: 1.1em; margin-left: 20px;">
           📸 View ${listing.images.length} Photos
          </button>`;
      listingsHtml += `</div>`;
    }
    
    // View property link
    if (listing.url && listing.url !== '#') {
      listingsHtml += `<div style="margin-top: 10px;">`;
      listingsHtml += `<a href="${listing.url}" target="_blank" style="color: var(--link); text-decoration: none; font-weight: 500;">🔗 View Full Details</a>`;
      listingsHtml += `</div>`;
    }
    
    listingsHtml += '</div>';
  });
  
  listingsHtml += '</div>';
  bubble.innerHTML = listingsHtml;
  row.appendChild(bubble);
  chatInner.appendChild(row);
  chatInner.scrollTop = chatInner.scrollHeight;
}

function showImagePopup(propertyId, images) {
  // Create modal overlay
  const modal = document.createElement('div');
  modal.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    z-index: 1000;
    display: flex;
    justify-content: center;
    align-items: center;
  `;
  
  // Create modal content
  const modalContent = document.createElement('div');
  modalContent.style.cssText = `
    background: var(--bg);
    border-radius: 10px;
    max-width: 95%;
    width: 1400px;
    max-height: 90%;
    padding: 20px;
    position: relative;
    overflow-y: auto;
  `;
  
  // Close button
  const closeBtn = document.createElement('button');
  closeBtn.innerHTML = '✕';
  closeBtn.style.cssText = `
    position: absolute;
    top: 10px;
    right: 15px;
    background: none;
    border: none;
    font-size: 24px;
    cursor: pointer;
    color: var(--text);
    z-index: 1001;
  `;
  closeBtn.onclick = () => document.body.removeChild(modal);
  
  // Title
  const title = document.createElement('h3');
  title.textContent = `Property Images (${images.length} photos)`;
  title.style.cssText = `
    margin: 0 0 20px 0;
    color: var(--text);
    text-align: center;
  `;
  
  // Images container
  const imagesContainer = document.createElement('div');
  imagesContainer.style.cssText = `
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 15px;
    max-height: 70vh;
    overflow-y: auto;
  `;
  
  // Add images
  images.forEach((imageUrl, index) => {
    const imgWrapper = document.createElement('div');
    imgWrapper.style.cssText = `
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    `;
    
    const img = document.createElement('img');
    img.src = imageUrl;
    img.alt = `Property Image ${index + 1}`;
    img.style.cssText = `
      width: 100%;
      height: 200px;
      object-fit: cover;
      cursor: pointer;
      transition: transform 0.2s;
    `;
    
    // Full size image on click
    img.onclick = () => {
      const fullSizeModal = document.createElement('div');
      fullSizeModal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.95);
        z-index: 1002;
        display: flex;
        justify-content: center;
        align-items: center;
      `;
      
      const fullImg = document.createElement('img');
      fullImg.src = imageUrl;
      fullImg.style.cssText = `
        max-width: 95%;
        max-height: 95%;
        object-fit: contain;
      `;
      
      fullSizeModal.onclick = () => document.body.removeChild(fullSizeModal);
      fullSizeModal.appendChild(fullImg);
      document.body.appendChild(fullSizeModal);
    };
    
    // Hover effect
    img.onmouseenter = () => img.style.transform = 'scale(1.05)';
    img.onmouseleave = () => img.style.transform = 'scale(1)';
    
    imgWrapper.appendChild(img);
    imagesContainer.appendChild(imgWrapper);
  });
  
  modalContent.appendChild(closeBtn);
  modalContent.appendChild(title);
  modalContent.appendChild(imagesContainer);
  modal.appendChild(modalContent);
  
  // Close on overlay click
  modal.onclick = (e) => {
    if (e.target === modal) {
      document.body.removeChild(modal);
    }
  };
  
  document.body.appendChild(modal);
}

async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text) return;
  
  addMessage('user', text);
  messageInput.value = '';
  
  const loadingId = 'loading-' + Date.now();
  addMessage('assistant', "...", loadingId);
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s safety timeout

    const response = await fetch('/api/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        chat_id: currentChatId,
        store_history: true
      }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    // Validate HTTP status first
    if (!response.ok) {
      const raw = (await response.text().catch(() => '')) || '';
      throw new Error(`HTTP ${response.status} ${response.statusText}${raw ? ` — ${raw.slice(0, 200)}` : ''}`);
    }

    // Ensure JSON content-type
    const ctype = (response.headers.get('content-type') || '').toLowerCase();
    if (!ctype.includes('application/json')) {
      const raw = (await response.text().catch(() => '')) || '';
      throw new Error(`Non-JSON response${raw ? ` — ${raw.slice(0, 200)}` : ''}`);
    }

    // Parse JSON
    const data = await response.json();
    
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) loadingElement.remove();
    
    if (data.error) {
      addMessage('assistant', `❌ Error: ${data.error}`);
    } else {
      currentChatId = data.chat_id;
      addMessage('assistant', data.assistant_message.content);
      if (data.assistant_message.meta && data.assistant_message.meta.listings) {
        displayPropertyListings(data.assistant_message.meta.listings);
      }
    }
  } catch (error) {
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) loadingElement.remove();
    addMessage('assistant', `❌ Network error: ${error && error.message ? error.message : 'Request failed'}`);
  }
}

// Event listeners
sendBtn.onclick = sendMessage;

messageInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Welcome message
window.addEventListener('load', () => {
  const welcomeMessage = `🏠 Welcome to Rihayish Chat Assistant!
  
I can help you find properties, answer questions about real estate listings, and provide detailed property information. Try asking me:
• "Show me houses under 50 lakh"
• "Find 3 bedroom houses in Usman Block"
• "What's the average price of houses?"

How can I assist you today?`;
  addMessage('assistant', welcomeMessage);
});
